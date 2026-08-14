# xspatula_ai4sh

Python package for seeding a PostgreSQL database with the AI4SoilHealth schema and reference data.

**Documentation**: [xspatula.github.io/xspatula_ai4sh_docs](https://xspatula.github.io/xspatula_ai4sh_docs/)

---

## What is this?

`xspatula_ai4sh` builds a PostgreSQL database for the [AI4SoilHealth](https://ai4soilhealth.eu) project using the Xspatula framework. All execution logic lives in JSON files — not in code. You define schemes, jobs, pilots, and processes in JSON; a Jupyter notebook calls the framework; the framework reads the JSON and runs the pipeline.

The database stores soil health observations, field measurements, spectral data, taxonomic records, and community/user management. It is the backbone for field data collected with the Xspectre pocket laboratory and linked earth observation data.

---

## Quick start

### 1. Create the Anaconda environment

```bash
conda env create --file anaconda/xspatula_py_3.12.yml
```

### 2. Edit the scheme file

Copy and edit one of the example scheme files in `setup/zzz/`:

| File | Purpose |
|---|---|
| `scheme_ai4sh_local_setup.json` | Create a new database |
| `scheme_ai4sh_local_delete.json` | Delete a database |
| `scheme_ai4sh_local_use.json` | Use an existing database |

Set the host, port, database name, and superuser credentials (or point to a `.netrc` entry).

### 3. Run the setup notebook

Open `setup/setup_db.ipynb` in VS Code or Jupyter, select the `xspatula_py_3.12` kernel, point to your scheme file, and run all cells.

To delete the database, use `setup/delete_db.ipynb` instead.

---

## Repository structure

```
xspatula_ai4sh/
├── src/
│   ├── lib/                  # Core framework library
│   │   ├── initiate.py       # Database and session initialisation
│   │   ├── login.py          # User authentication against the DB
│   │   ├── pilot.py          # Pilot file execution engine
│   │   ├── structure.py      # Schema and table structure management
│   │   └── version.py
│   ├── postgres/             # PostgreSQL connection and query helpers
│   │   ├── pg_session.py     # psycopg2 connection management
│   │   ├── pg_common.py      # Generic SQL helpers
│   │   ├── pg_processes.py   # Process table management
│   │   └── environment/      # Per-role .env credential files
│   └── utils/                # Shared utilities
│       ├── json_read_write.py
│       ├── code_log.py
│       ├── datumtid.py
│       ├── pretty_print.py
│       ├── struct.py
│       └── update_dict.py
├── setup/
│   ├── setup_db.ipynb        # Notebook: create database
│   ├── delete_db.ipynb       # Notebook: delete database
│   ├── setup_processes.ipynb # Notebook: register processes
│   ├── src_setup/            # Setup-specific Python source
│   │   └── lib_setup/        # DB setup, process management, privilege control
│   └── zzz/                  # Example scheme, job, pilot and process files
│       ├── scheme_ai4sh_local_setup.json
│       ├── scheme_ai4sh_local_delete.json
│       ├── scheme_ai4sh_local_use.json
│       └── ai4sh/            # AI4SoilHealth job, pilot and process files
└── anaconda/
    └── xspatula_py_3.12.yml  # Conda environment definition
```

---

## How it works

Every run follows a four-level hierarchy of JSON files:

```
scheme file  →  job file  →  pilot file  →  process file(s)
```

**Scheme file** — defines the database connection, superuser credentials, PostgreSQL users, and a pointer to the job file.

**Job file** — defines which pilot file (or process files) to run and where to find them.

**Pilot file** — a plain-text list of process files; lines starting with `#` are comments and are ignored, making it easy to toggle steps on/off.

**Process file** — a JSON array of operations, each with a `process_id` and `parameters`. Example:

```json
{
  "process": [
    {
      "process_id": "create_schema",
      "parameters": { "schema": "observation" }
    }
  ]
}
```

---

## Security model

Credentials are kept out of code using two mechanisms:

- **`.netrc`** — store PostgreSQL login credentials in `~/.netrc` and reference them by a machine code in the scheme file.
- **`.env` files** — generated automatically during setup; one per user category, stored in `src/postgres/environment/`. Used at runtime to open the database with the correct privileges.

Eight built-in PostgreSQL user roles are created during setup:

| Role | Purpose |
|---|---|
| `community_admin` | Manage users and organisations |
| `login_evaluation` | Validate login attempts (minimal rights) |
| `user_cat_0` – `user_cat_5` | Data access, most restricted (0) to most permissive (5) |

---

## Database schemas

A freshly seeded AI4SoilHealth database contains:

| Schema | Key tables |
|---|---|
| `utility` | Territory codes and shared lookup tables |
| `community` | `organisation`, `user`, `user_categories`, `user_media`, `user_activity` |
| `process` | `root_process`, `process`, `process_parameter`, parameter constraints and defaults |
| `observation` | Field observations and measurements, including eDNA (metabarcoding, taxa bioinformatics) |
| `observation_utility` | Lookup tables: indicators, units, methods, apparatus, taxa, spectroscopy, storage, eDNA sequencing/extraction/amplification methods, and more |
| `landscape` | Landscape observations and utility |

---

## Documentation

Full documentation at **[xspatula.github.io/xspatula_ai4sh_docs](https://xspatula.github.io/xspatula_ai4sh_docs/)**:

- Framework architecture — scheme files, job files, pilot files, process files, notebook interface
- Database setup — PostgreSQL, Anaconda, `.netrc`, schemas and tables

---

## Licenses

- **Code**: [MIT License](LICENSE)
- **Data**: [Creative Commons Attribution (CC-BY)](https://creativecommons.org/licenses/by/4.0/)
