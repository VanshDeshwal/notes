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
