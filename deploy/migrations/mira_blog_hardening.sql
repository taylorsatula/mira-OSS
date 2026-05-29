-- Harden existing mira_blog database privileges.
--
-- Run against the mira_blog database as a PostgreSQL superuser:
--   psql -U postgres -h localhost -d mira_blog -f deploy/migrations/mira_blog_hardening.sql
--
-- This migration does not set passwords. Set them after running:
--   ALTER ROLE mira_blog_reader WITH PASSWORD '...';
--   ALTER ROLE mira_blog_writer WITH PASSWORD '...';

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mira_blog_owner') THEN
        CREATE ROLE mira_blog_owner NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mira_blog_reader') THEN
        CREATE ROLE mira_blog_reader LOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mira_blog_writer') THEN
        CREATE ROLE mira_blog_writer LOGIN;
    END IF;
END
$$;

REVOKE CONNECT ON DATABASE mira_blog FROM PUBLIC;
GRANT CONNECT ON DATABASE mira_blog TO mira_blog_reader, mira_blog_writer;

REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO mira_blog_reader, mira_blog_writer;
GRANT ALL ON SCHEMA public TO mira_blog_owner;

ALTER TABLE posts OWNER TO mira_blog_owner;
ALTER SEQUENCE posts_id_seq OWNER TO mira_blog_owner;

REVOKE ALL ON posts FROM PUBLIC;
GRANT SELECT ON posts TO mira_blog_reader;
GRANT SELECT, INSERT, UPDATE, DELETE ON posts TO mira_blog_writer;

REVOKE ALL ON SEQUENCE posts_id_seq FROM PUBLIC;
GRANT USAGE, SELECT ON SEQUENCE posts_id_seq TO mira_blog_writer;

ALTER DEFAULT PRIVILEGES FOR ROLE mira_blog_owner IN SCHEMA public
    GRANT SELECT ON TABLES TO mira_blog_reader;

ALTER DEFAULT PRIVILEGES FOR ROLE mira_blog_owner IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO mira_blog_writer;

ALTER DEFAULT PRIVILEGES FOR ROLE mira_blog_owner IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO mira_blog_writer;
