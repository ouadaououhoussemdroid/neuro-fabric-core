-- Role enum
CREATE TYPE public.app_role AS ENUM ('individual', 'researcher', 'enterprise');

-- Profiles table
CREATE TABLE public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  role public.app_role NOT NULL DEFAULT 'individual',
  full_name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.profiles TO authenticated;
GRANT ALL ON public.profiles TO service_role;
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users view own profile" ON public.profiles FOR SELECT TO authenticated USING (auth.uid() = id);
CREATE POLICY "Users insert own profile" ON public.profiles FOR INSERT TO authenticated WITH CHECK (auth.uid() = id);
CREATE POLICY "Users update own profile" ON public.profiles FOR UPDATE TO authenticated USING (auth.uid() = id);

-- Researcher profile
CREATE TABLE public.researcher_profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL UNIQUE REFERENCES public.profiles(id) ON DELETE CASCADE,
  institution_name TEXT NOT NULL,
  publication_url TEXT,
  research_field TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.researcher_profiles TO authenticated;
GRANT ALL ON public.researcher_profiles TO service_role;
ALTER TABLE public.researcher_profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users view own researcher" ON public.researcher_profiles FOR SELECT TO authenticated USING (auth.uid() = user_id);
CREATE POLICY "Users insert own researcher" ON public.researcher_profiles FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users update own researcher" ON public.researcher_profiles FOR UPDATE TO authenticated USING (auth.uid() = user_id);

-- Enterprise profile
CREATE TABLE public.enterprise_profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL UNIQUE REFERENCES public.profiles(id) ON DELETE CASCADE,
  company_name TEXT NOT NULL,
  company_size TEXT,
  website TEXT,
  industry TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.enterprise_profiles TO authenticated;
GRANT ALL ON public.enterprise_profiles TO service_role;
ALTER TABLE public.enterprise_profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users view own enterprise" ON public.enterprise_profiles FOR SELECT TO authenticated USING (auth.uid() = user_id);
CREATE POLICY "Users insert own enterprise" ON public.enterprise_profiles FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users update own enterprise" ON public.enterprise_profiles FOR UPDATE TO authenticated USING (auth.uid() = user_id);