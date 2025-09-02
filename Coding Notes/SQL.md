- Write a query to calculate the average salary across all companies combined. Rename the column as `avg_salary`.

```sql
SELECT AVG(salary) AS avg_salary
FROM Works
```

- Write a query to retrieve the `department_name` and `location` of people who live in location that starts with 'S'.

```sql
SELECT department_name, location FROM departments
WHERE location LIKE 'S%'
```

- Write a query to select all the distinct companies (`company_name`) in the `Works` table..
```sql
SELECT DISTINCT company_name FROM Works
```

- Write a query to find the total count of books whose genre is **Fiction**.  
**Note**: Output column name should be `fiction_count`.

```sql
SELECT COUNT(*) as fiction_count FROM Books
WHERE genre = 'Fiction'
```

- Write a query to retrieve `book_id`, `title`, `author` and `published_year` of the books which have **NULL** rating for their books.

```sql
SELECT book_id, title, author, published_year FROM Library
WHERE rating IS NULL
```

- Create a query to retrieve the `employee_name`, `company`, and `salary` for employees in the full-time category, ordered by salary in **descending** order

```sql
SELECT employee_name, company, salary FROM Employees
WHERE category = 'Full-Time'
ORDER BY salary DESC
```

- Write a query to group the employees by their department and display the total number of employees (as total_employees) in each department.

```sql
SELECT department, COUNT(*) as total_employees FROM Employees 
GROUP BY department
```

- Write a query to retrieve the author_id, author_name, and publication_name for authors whose articles got zero views. The result should be sorted by author_id in ascending order.

```sql
SELECT author_id, author_name, publication_name FROM Views
WHERE view_count = 0
ORDER BY author_id
```

