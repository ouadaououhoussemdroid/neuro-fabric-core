import { createFileRoute, redirect } from "@tanstack/react-router";
import { supabase } from "@/integrations/supabase/client";

export const Route = createFileRoute("/_authenticated/dashboard/")({
  head: () => ({ meta: [{ title: "Dashboard · NeuroWeave" }] }),
  beforeLoad: async () => {
    const { data: userData } = await supabase.auth.getUser();
    const user = userData.user;
    if (!user) throw redirect({ to: "/signin" });
    const { data: profile } = await supabase
      .from("profiles")
      .select("role")
      .eq("id", user.id)
      .maybeSingle();
    const role = (profile?.role ?? "individual") as "individual" | "researcher" | "enterprise";
    if (role === "researcher") throw redirect({ to: "/dashboard/researcher" });
    if (role === "enterprise") throw redirect({ to: "/dashboard/enterprise" });
    throw redirect({ to: "/dashboard/individual" });
  },
  component: () => null,
});
