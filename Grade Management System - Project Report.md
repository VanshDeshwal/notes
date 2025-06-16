by- Vansh Deshwal (CrS2410)

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Project Structure](#2-project-structure)
- [3. Features Overview](#3-features-overview)
- [4. User Roles and Permissions](#4-user-roles-and-permissions)
- [5. Authentication & Security](#5-authentication--security)
- [6. Database Integration](#6-database-integration)
- [7. Detailed File Documentation](#7-detailed-file-documentation)
- [8. Theming and UI](#8-theming-and-ui)
- [9. Error Handling & Alerts](#9-error-handling--alerts)
- [10. Testing & Utilities](#10-testing--utilities)
- [11. Extensibility & Best Practices](#11-extensibility--best-practices)
- [12. Conclusion](#12-conclusion)

---

## 1. Introduction

The **Grade Management System** is a web-based application designed to facilitate the management of academic grades, courses, users, and audit logs for an educational institution. It supports three main user roles: **Admin**, **Professor**, and **Student**, each with tailored dashboards and permissions.

---

## 2. Project Structure

```
.
├── admin/
│   ├── audit_log.php
│   ├── dashboard.php
│   └── edit-grade.php
├── assets/
│   ├── css/
│   │   └── theme.css
│   └── js/
│       └── theme.js
├── config/
│   └── config.php
├── includes/
│   ├── alerts.php
│   ├── auth.php
│   ├── footer.php
│   ├── header.php
│   ├── nav_admin.php
│   ├── nav_professor.php
│   └── nav_student.php
├── lib/
│   └── db_helpers.php
├── professor/
│   ├── dashboard.php
│   ├── get_assessments.php
│   ├── get_enrolled_students.php
│   └── get_enrolled_students.php.new
├── student/
│   ├── dashboard.php
│   └── profile.php
├── .vscode/
│   └── settings.json
├── forgot_password.php
├── index.php
├── login.php
├── logout.php
├── test_db.php
└── test.html
```

---

## 3. Features Overview

- **User Authentication**: Secure login, session management, and role-based access.
- **Admin Dashboard**: Manage courses, view audit logs, and edit grades.
- **Professor Dashboard**: View assigned courses, submit/update grades, and view enrolled students and assessments.
- **Student Dashboard**: View grades, assessment breakdowns, and export grades as PDF.
- **Audit Logging**: Track all grade changes and administrative actions.
- **Responsive UI**: Bootstrap 5-based interface with light/dark theme toggle.
- **Database Integration**: MySQL backend with helper functions for common queries.
- **Error Handling**: User-friendly alerts and error messages.

---

## 4. User Roles and Permissions

- **Admin**
  - Full access to all data and actions.
  - Can view and edit any grade.
  - Can view audit logs.
- **Professor**
  - Can view and manage only their assigned courses.
  - Can submit and update grades for students in their courses.
- **Student**
  - Can view their own grades and profile.
  - Can update their profile and password.

---

## 5. Authentication & Security

- **Session Management**: All pages require authentication via session.
- **Role Checks**: Each page checks the user's role before granting access.
- **Password Hashing**: Passwords are securely hashed using PHP's `password_hash`.
- **Prepared Statements**: All database queries use prepared statements to prevent SQL injection.
- **Output Sanitization**: All user-facing content is sanitized using `htmlspecialchars`.

---

## 6. Database Integration & Structure

- **Configuration**: Database connection is set up in [`config/config.php`](config/config.php).
- **Helpers**: Common queries are abstracted in [`lib/db_helpers.php`](lib/db_helpers.php).
- **Testing**: [`test_db.php`](test_db.php) checks database connectivity and required tables.

### Users and Roles

The `users` table stores system users, categorized by roles: `admin`, `professor`, and `student`. Professors are linked to departments, and students are linked via `student_programs`.
### Academic Structure

- **Departments** are organized through the `departments` table.

- **Programs and Specializations** are captured using `programs`, `specializations`, and `student_programs`.

### Courses and Offerings

- **Courses** are defined in the `courses` table and linked to a department.

- **Offerings** in the `course_offerings` table connect courses to professors and semesters. 

### Assessment & Grading

- **Assessment Types** are listed in `assessment_types` with a defined weightage.

- **Course Assessments** are specific assessments per course offering (`course_assessments`).

- **Grades** are stored in `grades`, linking students, course offerings, and assessment instances.

## Entity Descriptions

### users
Holds all user data, including login credentials and roles.
### departments
Academic departments identified by a code and name.
### programs, specializations, student_programs
Defines educational programs, specializations, and tracks each student's enrollment.
### courses, course_offerings
Defines courses and which professor offers them in a specific term.
### assessment_types, course_assessments, grades
Defines types of assessments (e.g., Midsem, Endsem) and stores marks received by students.

## Database Connections and Relationships

### Primary Relationships

- `courses.department_id` → `departments.id`

- `course_offerings.course_id` → `courses.id`

- `course_offerings.professor_id` → `users.id`

- `course_assessments.course_offering_id` → `course_offerings.id`

- `course_assessments.assessment_type_id` → `assessment_types.id`

- `grades.student_id` → `users.id`

- `grades.course_offering_id` → `course_offerings.id`

- `grades.assessment_id` → `course_assessments.id`

- `student_programs.student_id` → `users.id`

- `student_programs.program_id` → `programs.id`

- `student_programs.specialization_id` → `specializations.id`

- `specializations.program_id` → `programs.id`

- `users.department_id` → `departments.id`


---

## 7. File Documentation

### Root Files

- **[`index.php`](index.php)**  
  Landing page. Redirects authenticated users to their respective dashboards.

- **[`login.php`](login.php)**  
  Handles user login, session creation, and redirects based on role.

- **[`logout.php`](logout.php)**  
  Destroys session and logs out the user.

- **[`forgot_password.php`](forgot_password.php)**  
  Disabled. Redirects to login.

- **[`test_db.php`](test_db.php)**  
  Tests database connection and checks for required tables.

- **[`test.html`](test.html)**  
  Static HTML page for testing UI and theme.

### Admin Panel ([admin/](admin/))

- **[`dashboard.php`](admin/dashboard.php)**  
  Displays current courses, enrolled students, and recent grade entries.

- **[`edit-grade.php`](admin/edit-grade.php)**  
  Allows admin to edit a specific grade, including marks, letter grade, and feedback. Updates audit log.

- **[`audit_log.php`](admin/audit_log.php)**  
  Shows a table of all audit log entries, including action, actor, entity, and description.

### Professor Panel ([professor/](professor/))

- **[`dashboard.php`](professor/dashboard.php)**  
  Main dashboard for professors. Lists assigned courses, allows grade submission/update, and displays course/assessment info.

- **[`get_enrolled_students.php`](professor/get_enrolled_students.php)**  
  Returns a JSON list of students enrolled in the selected course offering.

- **[`get_assessments.php`](professor/get_assessments.php)**  
  Returns a JSON list of assessment components for a course offering.

- **[`get_enrolled_students.php.new`](professor/get_enrolled_students.php.new)**  
  Alternate/new version of the enrolled students endpoint.

### Student Panel ([student/](student/))

- **[`dashboard.php`](student/dashboard.php)**  
  Displays all grades for the logged-in student, grouped by course and semester. Includes assessment breakdown and export to PDF.

- **[`profile.php`](student/profile.php)**  
  Allows students to view and update their profile and password.

### Includes ([includes/](includes/))

- **[`auth.php`](includes/auth.php)**  
  Session management and role-checking functions.

- **[`header.php`](includes/header.php)**  
  Common HTML `<head>` and opening `<body>` for all pages.

- **[`footer.php`](includes/footer.php)**  
  Common closing tags and scripts for all pages.

- **[`alerts.php`](includes/alerts.php)**  
  Displays Bootstrap toasts for session or local messages.

- **[`nav_admin.php`](includes/nav_admin.php)**  
  Navigation bar for admin users.

- **[`nav_professor.php`](includes/nav_professor.php)**  
  Navigation bar for professors.

- **[`nav_student.php`](includes/nav_student.php)**  
  Navigation bar for students.

### Assets ([assets/](assets/))

- **[`css/theme.css`](assets/css/theme.css)**  
  Custom CSS for theming, including light/dark mode and UI components.

- **[`js/theme.js`](assets/js/theme.js)**  
  JavaScript for toggling between light and dark themes.

### Config ([config/](config/))

- **[`config.php`](config/config.php)**  
  Database connection settings.

### Library ([lib/](lib/))

- **[`db_helpers.php`](lib/db_helpers.php)**  
  Helper functions for fetching students, courses, professors, course offerings, grades, departments, programs, specializations, and assessment types.

### VSCode Settings ([.vscode/](.vscode/))

- **[`settings.json`](.vscode/settings.json)**  
  Sets Live Server port for development.

---

## 8. Theming and UI

- **Bootstrap 5** is used for responsive layout and components.
- **Custom Theme**:  
  - Light and dark modes are supported.
  - Theme can be toggled via a button in the navbar.
  - Theme preference is saved in `localStorage`.
- **Accessibility**:  
  - Uses semantic HTML and ARIA labels.
  - Table captions and visually hidden elements for screen readers.

---

## 9. Error Handling & Alerts

- **Session-based Alerts**:  
  - Success and error messages are stored in `$_SESSION` and displayed as Bootstrap toasts.
- **Form Validation**:  
  - Both client-side (HTML5) and server-side validation for all forms.
- **Exception Handling**:  
  - Try/catch blocks for critical operations (e.g., grade editing).
  - User-friendly error messages.

---

## 10. Testing & Utilities

- **Database Test**:  
  - [`test_db.php`](test_db.php) checks connection and existence of required tables.
- **Static Test Page**:  
  - [`test.html`](test.html) for verifying UI and theme functionality.


