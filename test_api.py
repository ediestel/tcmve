#!/usr/bin/env python3
"""
Test script to verify API functionality and database saving.
"""

import requests
import json
import time

def test_api():
    """Test the TCMVE API with a simple query."""

    # Simple test query
    payload = {
        "query": "What is 2+2?",
        "flags": {
            "maxrounds": 1,
            "vice_check": True
        }
    }

    print("Testing TCMVE API...")
    print(f"Sending query: {payload['query']}")

    try:
        response = requests.post(
            "http://localhost:8000/api/run",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"Final answer: {result.get('final_answer', 'N/A')[:100]}...")
            print(f"Converged: {result.get('converged', False)}")
            print(f"Rounds: {result.get('rounds', 0)}")
            return True
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ API call error: {e}")
        return False

def check_database():
    """Check if the result was saved to database."""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        import os
        from dotenv import load_dotenv

        load_dotenv()

        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME", "tcmve"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "")
        )

        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute("SELECT COUNT(*) as count FROM runs")
            result = c.fetchone()
            count = result['count']
            print(f"📊 Database has {count} runs stored")

            if count > 0:
                c.execute("SELECT id, query, final_answer, created_at FROM runs ORDER BY id DESC LIMIT 1")
                latest = c.fetchone()
                print(f"📝 Latest run: ID {latest['id']}, Query: {latest['query'][:50]}...")
                print(f"   Created: {latest['created_at']}")
                return True
            else:
                print("⚠️  No runs found in database")
                return False

    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("TCMVE API & Database Test")
    print("=" * 40)

    # Test API
    api_success = test_api()

    # Wait a moment for database to sync
    time.sleep(2)

    # Check database
    db_success = check_database()

    print("\n" + "=" * 40)
    if api_success and db_success:
        print("🎉 SUCCESS: API working and results saving to database!")
    else:
        print("❌ ISSUES DETECTED:")
        if not api_success:
            print("  - API call failed")
        if not db_success:
            print("  - Database not saving results")