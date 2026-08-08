# Run this SQL in Neon dashboard to fix existing users timezone
# Go to: https://console.neon.tech → SQL Editor

UPDATE users SET timezone = 'Asia/Kolkata' WHERE timezone = 'UTC' OR timezone IS NULL;
UPDATE users SET briefing_time = '08:00' WHERE briefing_time IS NULL OR briefing_time = '';

-- Verify:
SELECT id, first_name, timezone, briefing_time, onboarded FROM users;
