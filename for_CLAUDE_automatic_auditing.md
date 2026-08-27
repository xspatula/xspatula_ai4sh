## Automatic building of audit system

On audit system, that you helped me build in a prevous session. 


## objective
I have made some changes in the database (related to eDNA - you can see from my commits) and I thus need to update the audit system. Other databases that I build would also benefit from an audit system and I want to create a generic system for building the audit system in an automatic manner. 

## idea

My idea is to extend the syntax for the process "create_table" used for defining all database tables (under @setup/zzz/ai4sh/setup_db/json_ai4sh/) and run from the notebook @setup/setup_db.ipynb. 

The file @setup/zzz/ai4sh/setup_db/json_ai4sh/utility/utility_v10_sql.json contains the example code I want to use for automatic creation of the audit system - it has an extra object "audit" with three
children "INSERT", "UPDATE" and "DELETE". 
  
```
{
  "process": [
    {
      "process_id": "create_table",
      "overwrite": false,
      "delete": false,
      "parameters": {
        "schema": "utility",
        "table": "foreign_key",
        "command": [
          "id SERIAL",
          "foreign_key TEXT",
          "dst_schema TEXT",
          "dst_table TEXT",
          "dst_search_column TEXT",
          "dst_alt_search_column TEXT",
          "PRIMARY KEY (foreign_key)"
        ]
      },
      "audit":{
        "INSERT": true,
        "UPDATE": true,
        "DELETE": true
      }
    },
```

So my idea is that when running @setup/setup_db.ipynb the audit system is built automatically. The basic audit json files (my guess which these are):
- @setup/zzz/ai4sh/setup_db/json_ai4sh/audit/audit_function_v10_sql.json
- @setup/zzz/ai4sh/setup_db/json_ai4sh/audit/audit_table_v10_sql.json
- @setup/zzz/ai4sh/setup_db/json_ai4sh/audit/audit_triggers_audit_v10_sql.json 

are built directly if they do not exists. If they do exists it means that @setup/setup_db.ipynb is run for an existing database and basic audit files should stay the same (I think). 

Looping over all the json files with the the object 
```
"process_id": "create_table",
```
python assembles all tables to be audited per schema and then writes the json command files for defining the auditing per schema (e.g. "audit_triggers_<schema>_vxx_sql.json"). 

Instead of running the audit defition JSON files from @setup/zzz/ai4sh/setup_db/db_xspatula_ai4sh_setup.txt, they are automatically run from the script (there are some comments you wrote before in @setup/zzz/ai4sh/setup_db/db_xspatula_ai4sh_setup.txt that hints at running all the triggers at the very end). By default, and not allowed to be changed, any existing record for auditing should only be allowed to be updated (not deleted) and the update should only happen if the new data is different from the old - this is important as @setup/setup_db.ipynb allows rerunning the whole database setup with all tables defined and if "overwrite" is set to false nothing happens (and nothing should happen in the audit tables), but if there is one new table, this is added. And that should be be added also in the auditing. If a table is deleted when running @setup/setup_db.ipynb, then the auditing can also be deleted (I think). I also think that a table that is updated ("overwrite": "true") when running @setup/setup_db.ipynb it is actually first deleted and then recreated - you need to check that and handle it for the auditing as this should probably be recorded as an update, Please check.

## 3 steps to achieve my idea

I envisions the following three steps to achieve what I want

### Step 1 - create the required code and test in parallel to existing system

Use the example I created (@setup/zzz/ai4sh/setup_db/json_ai4sh/utility/utility_v10_sql.json) to rewrite the code to handle the automatic auditing definition - including the writing of the 3 basic audit JSON files. If the schema "audit" does ot exists, ask the the user if he/she wnats an audit system, if "y" create the schema, if "n" skip the whole audit setup.

### Step 2 - set the audit definition in all JSON files

From the existing audit definitions (where you helped me define which tables to audit), rewrite all the JSON files with "process_id": "create_table" to have the new "audit" object reflect the existing settings.

### Step 3 - implement and test the automatic audit system

The third step would be to implement and test the automatic audit setup system and control that it gives results matching the existing system. 

