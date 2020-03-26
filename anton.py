def one():
    import sqlite3

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    cursor.execute('CREATE TABLE students (name text, family text, groups text, course int)')
    conn.commit()

    students = [('раз', 'разов', 'ПМ', 1),
                ('два', 'двоев', 'ИВТ', 1),
                ('три', 'троев', 'МО', 2)]

    cursor.executemany('INSERT INTO students VALUES (?, ?, ?, ?)', students)
    conn.commit()

    cursor.execute('SELECT * FROM students WHERE course=1')
    print(cursor.fetchall())

    cursor.execute('DELETE FROM students WHERE course=1')
    conn.commit()

    cursor.execute('SELECT * FROM students WHERE course=1')
    print(cursor.fetchall())


def two():
    import csv
    import sqlite3

    data = []
    with open('data.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    for i in range(len(data)):
        data[i][3] = int(data[i][3]) + 1
        data[i].append('false')

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    cursor.execute('CREATE TABLE students (name text, family text, groups text, course int, arrears text)')
    conn.commit()

    cursor.executemany('INSERT INTO students VALUES (?, ?, ?, ?, ?)', data)
    conn.commit()

    cursor.execute('SELECT * FROM students')
    print(cursor.fetchall())

    with open('save.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for line in data:
            writer.writerow(line)


while True:
    try:
        cmd = int(input('Choose 1-2: '))
    except ValueError:
        break
    tasks = [one, two]
    try:
        tasks[cmd-1]()
    except IndexError:
        pass
