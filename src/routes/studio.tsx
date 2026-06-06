import { createFileRoute } from "@tanstack/react-router";
import { SiteShell } from "@/components/site-shell";
import { Section, PageHeader } from "@/components/ui-bits";

export const Route = createFileRoute("/studio")({
  component: StudioPage,
});

function StudioPage() {
  return (
    <SiteShell>
      <Section>
        <PageHeader
          eyebrow="Neural Processing Studio"
          title="Coming Soon"
          sub="The full EEG studio is under development."
        />
        <div className="mt-10 text-center text-muted-foreground">
          Studio page is temporarily disabled for build stability.
        </div>
      </Section>
    </SiteShell>
  );
}
