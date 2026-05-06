import sqlite3, hashlib

DB = 'attendance.db'
conn = sqlite3.connect(DB)
cursor = conn.cursor()

cursor.execute('SELECT id, employee_id, username, password FROM faculty')
rows = cursor.fetchall()
updated = 0
for row in rows:
    fid, emp_id, username, pwd = row
    if len(pwd) != 64:  # plain text - not sha256
        login_id = emp_id if emp_id else username
        new_pass = hashlib.sha256(f'{login_id}@123'.encode()).hexdigest()
        cursor.execute('UPDATE faculty SET password=? WHERE id=?', (new_pass, fid))
        print(f'Fixed: {emp_id or username}  -> new password = {login_id}@123')
        updated += 1
    else:
        print(f'OK (already hashed): {emp_id or username}')

conn.commit()
conn.close()
print(f'\nDone. Fixed {updated} faculty record(s).')
