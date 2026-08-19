/**
 * Mission 15 Phase 1 — Creates real test users in local GoTrue + saves real JWTs.
 * No mocks: users created via real GoTrue HTTP API; JWTs signed by GoTrue.
 */
import crypto from "node:crypto";
import { createClient } from "@supabase/supabase-js";
import { writeFileSync } from "node:fs";

const API_URL = "http://127.0.0.1:54321";
const ANON_KEY = "sb_publicable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH";
const SERVICE_ROLE_KEY = process.env.M15_SERVICE_ROLE_KEY;
if (!SERVICE_ROLE_KEY) { console.error("Set M15_SERVICE_ROLE_KEY"); process.exit(1); }

const admin = createClient(API_URL, SERVICE_ROLE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false },
});
const JWT_SECRET = "super-secret-jwt-token-with-at-least-32-characters-long";

function b64url(buf) {
  return Buffer.from(buf).toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}
function makeJWT(payload) {
  const header = { alg: "HS256", typ: "JWT", kid: "pm" };
  const data = `${b64url(JSON.stringify(header))}.${b64url(JSON.stringify(payload))}`;
  return `${data}.${b64url(crypto.createHmac("sha256", JWT_SECRET).update(data).digest())}`;
}

async function ensureUser(email, password) {
  // Try sign-in first (user may already exist from a prior run)
  let { data: sd, error: se } = await admin.auth.signInWithPassword({ email, password });
  if (sd?.session?.access_token) {
    console.log(`[${email}] signed in (exists), id=${sd.user.id}`);
    return { id: sd.user.id, jwt: sd.session.access_token };
  }
  // Create the user if not found
  const { data: cd, error: ce } = await admin.auth.admin.createUser({
    email, password, email_confirm: true,
  });
  if (ce) throw ce;
  console.log(`[${email}] created user: ${cd.user.id}`);
  // Sign in to get a real JWT from GoTrue
  const { data: signData, error: signErr } = await admin.auth.signInWithPassword({ email, password });
  if (signErr) throw signErr;
  return { id: signData.user.id, jwt: signData.session.access_token };
}

const password = "M15TestPass!123";
const userA = await ensureUser("m15-usera@m15.test", password);
const userB = await ensureUser("m15-userb@m15.test", password);

// Expired JWT (exp in the past) — real GoTrue signing key, expired
const now = Math.floor(Date.now() / 1000);
const expiredJWT = makeJWT({
  iss: API_URL, role: "authenticated", sub: userA.id,
  email: "m15-usera@m15.test",
  exp: now - 3600, iat: now - 7200, aud: "authenticated",
});

// Valid-structure JWT with tampered signature (invalid signature)
const goodJWT = makeJWT({
  iss: API_URL, role: "authenticated", sub: userA.id,
  email: "m15-usera@m15.test",
  exp: now + 3600, iat: now, aud: "authenticated",
});
const parts = goodJWT.split(".");
parts[2] = "Z".repeat(43);
const invalidJWT = parts.join(".");

const tokens = {
  userA: { id: userA.id, email: "m15-usera@m15.test", jwt: userA.jwt },
  userB: { id: userB.id, email: "m15-userb@m15.test", jwt: userB.jwt },
  expiredJWT,
  invalidJWT,
  serviceRoleKey: SERVICE_ROLE_KEY,
  anonKey: ANON_KEY,
  jwtSecret: JWT_SECRET,
  apiUrl: API_URL,
  createdAt: new Date().toISOString(),
};

writeFileSync("reports/m15_jwt_test_tokens.json", JSON.stringify(tokens, null, 2));
console.log("=== Tokens saved ===");
console.log(`userA JWT len: ${userA.jwt.length}`);
console.log(`userB JWT len: ${userB.jwt.length}`);
console.log(`expiredJWT len: ${expiredJWT.length}`);
console.log(`invalidJWT len: ${invalidJWT.length}`);
