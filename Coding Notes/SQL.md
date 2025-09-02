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

Write a query to find the names of the **top 3 distinct players** by highest score who have **won** matches, including their **scores**.

Table 1: **Players**

|player_id|player_name|score|rank|
|---|---|---|---|
|1|Alice|1200|5|
|2|Bob|1500|2|
|3|Charlie|1300|4|
|4|David|1600|1|
|5|Eve|1100|6|

Table 2: **Matches**

|match_id|player1|player2|winner|match_date|
|---|---|---|---|---|
|101|Alice|Bob|Bob|2024-01-15|
|102|Charlie|David|David|2024-01-16|
|103|Eve|Bob|Bob|2024-01-17|
|104|Alice|David|David|2024-01-18|
|105|Charlie|Eve|Charlie|2024-01-19|
```sql
SELECT p.player_name, p.score
FROM Players p
WHERE p.player_name IN (
    SELECT winner
    FROM Matches
)
ORDER BY p.score DESC
LIMIT 3;
```

- Write a query to retrieve the details of the **last five matches** played, including the match ID, the names of the players who participated, the name of the winning player, match date, and the final score of the winner.

	- **Players table:** `player_id`, `player_name`, `score`, `rank`
	-  **Matches table:** `match_id`, `player1`, `player2`, `winner`, `match_date`

```sql
SELECT 
    m.match_id,                        -- Match ID
    m.player1,         -- Player 1 name
    m.player2,         -- Player 2 name
    m.winner,           -- Winner name
    m.match_date,                      -- Date of the match
    p.score            -- Winner's score
FROM Matches AS m                     -- (Take data from Matches table)
JOIN Players AS p                     -- (Join with Players table to get winner's score)
    ON m.winner = p.player_name       -- (Match winner name with player_name in Players)
ORDER BY 
    m.match_date DESC                 -- (Sort by date, latest first)
LIMIT 5;   
```