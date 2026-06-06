export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  __InternalSupabase: {
    PostgrestVersion: "14.5"
  }
  public: {
    Tables: {
      enterprise_profiles: {
        Row: {
          company_name: string
          company_size: string | null
          created_at: string
          id: string
          industry: string | null
          user_id: string
          website: string | null
        }
        Insert: {
          company_name: string
          company_size?: string | null
          created_at?: string
          id?: string
          industry?: string | null
          user_id: string
          website?: string | null
        }
        Update: {
          company_name?: string
          company_size?: string | null
          created_at?: string
          id?: string
          industry?: string | null
          user_id?: string
          website?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "enterprise_profiles_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: true
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      profiles: {
        Row: {
          avatar_url: string | null
          created_at: string
          full_name: string | null
          id: string
          role: Database["public"]["Enums"]["app_role"]
        }
        Insert: {
          avatar_url?: string | null
          created_at?: string
          full_name?: string | null
          id: string
          role?: Database["public"]["Enums"]["app_role"]
        }
        Update: {
          avatar_url?: string | null
          created_at?: string
          full_name?: string | null
          id?: string
          role?: Database["public"]["Enums"]["app_role"]
        }
        Relationships: []
      }
      researcher_profiles: {
        Row: {
          created_at: string
          id: string
          institution_name: string
          publication_url: string | null
          research_field: string | null
          user_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          institution_name: string
          publication_url?: string | null
          research_field?: string | null
          user_id: string
        }
        Update: {
          created_at?: string
          id?: string
          institution_name?: string
          publication_url?: string | null
          research_field?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "researcher_profiles_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: true
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      waitlist: {
        Row: {
          created_at: string
          email: string
          id: string
          name: string | null
        }
        Insert: {
          created_at?: string
          email: string
          id?: string
          name?: string | null
        }
        Update: {
          created_at?: string
          email?: string
          id?: string
          name?: string | null
        }
        Relationships: []
      }
      eeg_analyses: {
        Row: {
          id: string
          user_id: string
          file_name: string
          file_size_bytes: number
          sample_rate: number
          num_channels: number
          num_samples: number
          embedding: number[]
          embedding_dimensions: number
          embedding_model: string
          attention: number
          workload: number
          arousal: number
          bandpass_low: number | null
          bandpass_high: number | null
          notch_frequency: number | null
          processing_time_ms: number
          created_at: string
        }
        Insert: {
          id?: string
          user_id: string
          file_name: string
          file_size_bytes: number
          sample_rate: number
          num_channels: number
          num_samples: number
          embedding: number[]
          embedding_dimensions: number
          embedding_model: string
          attention: number
          workload: number
          arousal: number
          bandpass_low?: number | null
          bandpass_high?: number | null
          notch_frequency?: number | null
          processing_time_ms: number
          created_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          file_name?: string
          file_size_bytes?: number
          sample_rate?: number
          num_channels?: number
          num_samples?: number
          embedding?: number[]
          embedding_dimensions?: number
          embedding_model?: string
          attention?: number
          workload?: number
          arousal?: number
          bandpass_low?: number | null
          bandpass_high?: number | null
          notch_frequency?: number | null
          processing_time_ms?: number
          created_at?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      app_role: "individual" | "researcher" | "enterprise"
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">
type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export const Constants = {
  public: {
    Enums: {
      app_role: ["individual", "researcher", "enterprise"],
    },
  },
} as const
