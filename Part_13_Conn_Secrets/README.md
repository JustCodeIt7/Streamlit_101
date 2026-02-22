# Part 13: Connections and Secrets in Streamlit

This section covers secure configuration management, database connections, and secrets handling.

## Overview

Streamlit provides built-in support for:
- **Secrets Management**: Securely store API keys, passwords, and credentials
- **Connections**: Connect to databases and external services
- **Environment Variables**: Access system environment variables

## Files in This Section

- **[app.py](Part_13_Conn_Secrets/app.py)** - Main application demonstrating connections and secrets
- **[p13_app.py](Part_13_Conn_Secrets/p13_app.py)** - Tutorial script with SQL connection example
- **app2.py** - Alternative implementation
- **pets.db** - SQLite database file

## Key Concepts Covered

### 1. Accessing Secrets

Secrets are stored in `.streamlit/secrets.toml` and accessed via `st.secrets`:

```python
# Dictionary syntax
db_username = st.secrets["db_username"]

# Attribute syntax
db_password = st.secrets.db_password
```

**Key Points:**
- Secrets stored in `.streamlit/secrets.toml`
- Both dictionary and attribute access supported
- Never commit secrets to version control

---

### 2. Environment Variables

Access system environment variables:

```python
import os

# Check if env var matches secret
os.environ["db_username"] == st.secrets["db_username"]
```

**Use Cases:**
- Integration with existing infrastructure
- Docker/Kubernetes secrets
- CI/CD pipeline integration

---

### 3. SQL Connections

Create database connections using `st.connection()`:

```python
conn = st.connection("pets_db", type="sql")
```

**Key Points:**
- Connection name must match secrets configuration
- Use `type="sql"` for SQL databases
- Supports SQLite, PostgreSQL, MySQL, and more

---

### 4. Database Operations

Perform CRUD operations with connection sessions:

```python
from sqlalchemy import text

# Create table
with conn.session as s:
    s.execute(text("CREATE TABLE IF NOT EXISTS pet_owners (person TEXT, pet TEXT);"))
    s.commit()

# Insert data
with conn.session as s:
    s.execute(
        text("INSERT INTO pet_owners (person, pet) VALUES (:owner, :pet);"),
        params=dict(owner="jerry", pet="fish")
    )
    s.commit()

# Query data
pet_owners = conn.query("SELECT * FROM pet_owners", ttl=500)
st.dataframe(pet_owners)
```

**Key Points:**
- Use `text()` wrapper for SQL statements
- Use parameterized queries to prevent SQL injection
- `ttl` parameter sets cache time-to-live in seconds

---

### 5. Secrets Configuration File

Create `.streamlit/secrets.toml`:

```toml
[db_username]
# Database username

[db_password]
# Database password

[pets_db]
drivername = "sqlite"
database = "pets.db"
```

## Running the App

```bash
streamlit run Part_13_Conn_Secrets/app.py
# or
streamlit run Part_13_Conn_Secrets/p13_app.py
```

**Note**: Ensure `.streamlit/secrets.toml` is properly configured before running.

## Next Steps

After completing this section, you should understand:
- How to securely store and access secrets
- Creating database connections
- Performing SQL operations with sessions
- Using environment variables for configuration

Proceed to [Part 14: Cache](../Part_14_Cache/) to learn about performance optimization.