CREATE DATABASE IF NOT EXISTS spml_attendance
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE spml_attendance;

CREATE TABLE IF NOT EXISTS employee_attendance (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    employee_id VARCHAR(128) NOT NULL,
    attendance_date DATE NOT NULL,
    in_time DATETIME NOT NULL,
    out_time DATETIME NULL,
    status ENUM('IN', 'OUT') NOT NULL,
    entry_camera_id INT NOT NULL,
    exit_camera_id INT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_employee_date (employee_id, attendance_date),
    INDEX idx_open_sessions (employee_id, out_time)
);

-- Run separately with a strong local password, then grant only this schema.
-- CREATE USER 'spml_attendance_app'@'localhost' IDENTIFIED BY '<strong-password>';
-- GRANT SELECT, INSERT, UPDATE ON spml_attendance.* TO
--   'spml_attendance_app'@'localhost';
-- FLUSH PRIVILEGES;
