#!/usr/bin/env bun

/**
 * Phase 2C: Dataset population seed script for Neuro-Fabric's curated BCI datasets.
 *
 * This script populates the `datasets` table in Supabase with standardized metadata
 * for common BCI datasets used in the Neuro-Fabric platform.
 *
 * To run:
 *   bun run scripts/populate-datasets.ts
 *
 * Ensure the following environment variables are set:
 *   - SUPABASE_URL: The Supabase project URL
 *   - SUPABASE_SERVICE_ROLE_KEY: The service role key (for bypassing RLS)
 */

import { createClient } from "@supabase/supabase-js";

// Load environment variables
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseServiceRoleKey) {
  console.error("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set");
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceRoleKey);

// Define the datasets to seed
const datasets = [
  {
    name: "BCI-IV-2a",
    license: "BSD-3-Clause",
    source_url: "http://bnci-horizon-2020.eu/database/data-sets",
    n_subjects: 9,
    n_channels: 22,
    sample_rate: 250,
    n_classes: 4,
    preprocessing_sha256: "", // To be filled after preprocessing pipeline is run
    metadata: {
      paradigm: "Motor Imagery",
      description: "BCI Competition IV Dataset 2a: 4-class motor imagery task",
      version: "1.0.0",
    },
  },
  // Add more datasets as needed
  // Example:
  // {
  //   name: 'BNCI2014009',
  //   license: 'BSD-3-Clause',
  //   source_url: 'https://www.bncihorizon.org/html/wellknown/datasets/BNCI2014009.html',
  //   n_subjects: 8,
  //   n_channels: 16,
  //   sample_rate: 250,
  //   n_classes: 2,
  //   preprocessing_sha256: '',
  //   metadata: {
  //     paradigm: 'P300',
  //     description: 'BNCI Horizon 2014-009: P300 speller',
  //     version: '1.0.0',
  //   },
  // },
];

async function main() {
  console.log("Starting dataset population...");

  for (const dataset of datasets) {
    const { data, error } = await supabase.from("datasets").upsert(
      {
        name: dataset.name,
        license: dataset.license,
        source_url: dataset.source_url,
        n_subjects: dataset.n_subjects,
        n_channels: dataset.n_channels,
        sample_rate: dataset.sample_rate,
        n_classes: dataset.n_classes,
        preprocessing_sha256: dataset.preprocessing_sha256,
        metadata: dataset.metadata,
        // updated_at will be set automatically by the database trigger
      },
      { onConflict: ["name"] }, // Use name as the unique conflict target
    );

    if (error) {
      console.error(`Error upserting dataset ${dataset.name}:`, error);
      process.exit(1);
    }

    console.log(`✓ Upserted dataset: ${dataset.name}`);
  }

  console.log("Dataset population completed successfully.");
}

// Run the main function and handle errors
main().catch((error) => {
  console.error("Unexpected error:", error);
  process.exit(1);
});
