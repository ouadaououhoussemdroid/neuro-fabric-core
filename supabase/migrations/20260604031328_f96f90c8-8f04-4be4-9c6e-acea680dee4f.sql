
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  v_role public.app_role;
  v_full_name text;
  v_meta jsonb;
begin
  v_meta := coalesce(new.raw_user_meta_data, '{}'::jsonb);
  v_full_name := nullif(v_meta->>'full_name', '');
  v_role := coalesce(nullif(v_meta->>'role', '')::public.app_role, 'individual'::public.app_role);

  insert into public.profiles (id, role, full_name)
  values (new.id, v_role, v_full_name)
  on conflict (id) do nothing;

  if v_role = 'researcher' and coalesce(v_meta->>'institution_name','') <> '' then
    insert into public.researcher_profiles (user_id, institution_name, publication_url, research_field)
    values (
      new.id,
      v_meta->>'institution_name',
      nullif(v_meta->>'publication_url',''),
      nullif(v_meta->>'research_field','')
    );
  elsif v_role = 'enterprise' and coalesce(v_meta->>'company_name','') <> '' then
    insert into public.enterprise_profiles (user_id, company_name, company_size, website, industry)
    values (
      new.id,
      v_meta->>'company_name',
      nullif(v_meta->>'company_size',''),
      nullif(v_meta->>'website',''),
      nullif(v_meta->>'industry','')
    );
  end if;

  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();
