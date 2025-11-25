-- TCMVE Database Initialization
-- PostgreSQL setup script for Docker

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Grant permissions to tcmve user
GRANT ALL PRIVILEGES ON DATABASE tcmve TO tcmve;

-- Note: Tables will be created by setup_database.py when the application starts
-- This file ensures the database is ready for TCMVE initialization