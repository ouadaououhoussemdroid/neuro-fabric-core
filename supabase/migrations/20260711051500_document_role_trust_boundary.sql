-- NFC-013: document the review outcome for client-controlled role
-- self-assignment at signup, rather than changing behavior.
--
-- handle_new_user() trusts raw_user_meta_data->>'role' from the
-- client's signup payload. Reviewed whether this lets a user
-- self-assign elevated privilege:
--   - No RLS policy anywhere in this schema conditions on
--     profiles.role (grep-verified across all migrations).
--   - The only application-code use of profile.role is UI routing
--     (which /dashboard/* variant to redirect to) — every dashboard
--     queries the same auth.uid()-scoped tables regardless of role.
--   - researcher_profiles.institution_name / enterprise_profiles.company_name
--     are self-reported free text, displayed as-is with no "verified"
--     claim attached anywhere in the UI.
-- Conclusion: self-assigning "researcher"/"enterprise" changes which
-- dashboard UI a user sees and what self-reported profile text is
-- stored, but grants no elevated data access or privilege. No code
-- change made; this is metadata for the next reviewer so the question
-- doesn't have to be re-derived from scratch.
COMMENT ON FUNCTION public.handle_new_user() IS
  'Trusts client-supplied role (individual|researcher|enterprise) from '
  'signup metadata. Reviewed 2026-07-11 (NFC-013): role is not referenced '
  'by any RLS policy and gates only dashboard UI routing, not data access '
  '— self-assignment is not a privilege-escalation vector. If a future '
  'feature ever makes role security-relevant, this trust boundary must be '
  'revisited.';
