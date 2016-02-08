import ballpark


def table(data, spacing=2):
    columns = [[key] + ballpark.business(column) for key, column in data]
    rows = zip(*columns)
    widths = [max(map(len, column)) for column in columns]
    spacing = ' ' * spacing

    lines = []
    for row in rows:
        cells = ['{value:>{width}}'.format(value=cell, width=widths[ix]) for ix, cell in enumerate(row)]
        lines.append(spacing.join(cells))

    return '\n'.join(lines)


data = {
    'clouds': [100.111, 200.2222, 300.333], 
    'rainfall': [5.2342, 9.99812, 2.1],
    'sun': [131311122.270, 1021221.112, 2331004.005],
}

print(table(data.items()))
