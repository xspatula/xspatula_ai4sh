## Using PgAdmin

## List schemas in database

```
SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN 
('information_schema', 'pg_catalog','pg_toast');
```

## List tables in schema

```
SELECT * FROM information_schema.tables 
WHERE table_schema = 'process';
``` 

