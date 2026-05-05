-- Migration: v1 → v2
-- Adds instrument lookup table columns to an existing afm_scans table.
-- Safe to run multiple times (ADD COLUMN IF NOT EXISTS is idempotent).
--
-- Run against your Nautilus postgres pod:
--   kubectl exec -it <postgres-pod> -n gvelard2 -- \
--     psql -U afm -d afm -f /dev/stdin < deploy/migrate_v2.sql

ALTER TABLE afm_scans ADD COLUMN IF NOT EXISTS scan_rate_hz       REAL;
ALTER TABLE afm_scans ADD COLUMN IF NOT EXISTS scan_angle_deg     REAL;
ALTER TABLE afm_scans ADD COLUMN IF NOT EXISTS scan_lines         INT;
ALTER TABLE afm_scans ADD COLUMN IF NOT EXISTS scan_points        INT;
ALTER TABLE afm_scans ADD COLUMN IF NOT EXISTS drive_frequency_hz REAL;
ALTER TABLE afm_scans ADD COLUMN IF NOT EXISTS drive_amplitude_v  REAL;
ALTER TABLE afm_scans ADD COLUMN IF NOT EXISTS spring_constant    REAL;
ALTER TABLE afm_scans ADD COLUMN IF NOT EXISTS tip_voltage_v      REAL;
ALTER TABLE afm_scans ADD COLUMN IF NOT EXISTS instrument_model   TEXT;
ALTER TABLE afm_scans ADD COLUMN IF NOT EXISTS scan_date          TEXT;
