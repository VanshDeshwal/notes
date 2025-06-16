

Follow these steps to install and set up the Project

---

## 1. Prerequisites

- **XAMPP** (or any Apache + MySQL + PHP stack)
- **Web browser**

---

## 2. Download  the Project Files

- Download the project folder and extract it to your computer.

---

## 3. Place Project in Web Server Directory

- Move the project folder (e.g., `grade-management-system`) to your web server's root directory:
  - For XAMPP: `C:\xampp\htdocs\grade-management-system`
  - For Linux: `/var/www/html/grade-management-system`

---

## 4. Import the Database

1. Open **phpMyAdmin** (http://localhost/phpmyadmin).
2. Create a new database named `grades_db`.
3. Import the provided SQL schema and data:
   - Click the `Import` tab in phpMyAdmin.
   - Select the SQL file (e.g., `grades_db.sql`) from your project.
   - Click **Go** to import.

---

## 5. Configure Database Connection

- Open [`config/config.php`](config/config.php).
- Ensure the following settings match your MySQL setup:
  ```php
  $host = "localhost";
  $db = "grades_db";
  $user = "root";
  $pass = ""; // Default for XAMPP
  ```

---

## 6. Start Apache and MySQL

- Open the **XAMPP Control Panel**.
- Start **Apache** and **MySQL**.

---

## 7. Access the Application

- Open your browser and go to:
  ```
  http://localhost/grade-management-system/
  ```

---

## 8. Test the Setup

- Visit [`test_db.php`](test_db.php) to verify database connectivity:
  ```
  http://localhost/grade-management-system/test_db.php
  ```
- You should see messages about database connection and table status.

---

## 9. Default Logins

- Use the credentials you set up in the `users` table.
- currently default password for all users is `pass123`

---

## 10. Troubleshooting

- If you see database errors, double-check:
  - MySQL is running.
  - Database credentials in [`config/config.php`](config/config.php) are correct.
  - The database and tables exist.
- For PHP errors, ensure all required PHP extensions are enabled in XAMPP.

---

## 11. Optional: Enable Live Server (VSCode)

- If using VSCode, you can use the Live Server extension.
- The port is set in [`.vscode/settings.json`](.vscode/settings.json).
