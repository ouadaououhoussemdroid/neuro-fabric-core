import { useState, ReactNode } from "react";
import { z } from "zod";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { supabase } from "@/integrations/supabase/client";
import { CheckCircle2, Loader2 } from "lucide-react";

const schema = z.object({
  email: z.string().trim().email("Enter a valid email").max(255),
  name: z.string().trim().max(100).optional().or(z.literal("")),
});

export function WaitlistDialog({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    setEmail("");
    setName("");
    setStatus("idle");
    setError(null);
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const parsed = schema.safeParse({ email, name });
    if (!parsed.success) {
      setError(parsed.error.issues[0]?.message ?? "Invalid input");
      return;
    }
    setStatus("loading");
    const { error: dbError } = await supabase.from("waitlist").insert({
      email: parsed.data.email,
      name: parsed.data.name ? parsed.data.name : null,
    });
    if (dbError) {
      setStatus("error");
      if (dbError.code === "23505") {
        setError("This email is already on the waitlist.");
      } else {
        setError(dbError.message);
      }
      return;
    }
    setStatus("success");
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        setOpen(o);
        if (!o) setTimeout(reset, 200);
      }}
    >
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent className="border-border/60 bg-background/95 backdrop-blur-xl sm:max-w-md">
        {status === "success" ? (
          <div className="flex flex-col items-center gap-3 py-6 text-center">
            <div className="grid h-12 w-12 place-items-center rounded-full bg-neuro-gradient glow">
              <CheckCircle2 className="h-6 w-6 text-background" />
            </div>
            <DialogTitle className="text-xl">You're on the list</DialogTitle>
            <DialogDescription className="max-w-sm">
              Thanks for requesting access to NeuroWeave. We'll reach out at{" "}
              <span className="font-mono text-foreground">{email}</span> as cohorts open up.
            </DialogDescription>
          </div>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>Request access</DialogTitle>
              <DialogDescription>
                Join the NeuroWeave private beta. We onboard new researchers and teams weekly.
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={onSubmit} className="mt-2 space-y-4">
              <div className="space-y-2">
                <Label htmlFor="wl-email">Email</Label>
                <Input
                  id="wl-email"
                  type="email"
                  required
                  maxLength={255}
                  placeholder="you@lab.org"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={status === "loading"}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="wl-name">
                  Name <span className="text-muted-foreground">(optional)</span>
                </Label>
                <Input
                  id="wl-name"
                  type="text"
                  maxLength={100}
                  placeholder="Ada Lovelace"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={status === "loading"}
                />
              </div>
              {error && (
                <p className="text-sm text-destructive" role="alert">
                  {error}
                </p>
              )}
              <button
                type="submit"
                disabled={status === "loading"}
                className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-neuro-gradient px-4 py-2.5 text-sm font-medium text-background glow transition-transform hover:scale-[1.01] disabled:opacity-60"
              >
                {status === "loading" ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" /> Submitting…
                  </>
                ) : (
                  "Request access"
                )}
              </button>
              <p className="text-center text-[11px] text-muted-foreground">
                We only use your email to contact you about access.
              </p>
            </form>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
