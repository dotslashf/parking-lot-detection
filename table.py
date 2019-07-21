from terminaltables import SingleTable

data = []

data.append(['Sasa', 'Fadhlu'])
data.append(['Jelek', 'Ganteng'])
data.append(['Jelek Sekali', 'Ganteng Sekali'])

table = SingleTable(data)
table.title = "Tabel Kesayanganku"
table.inner_row_border = True

print (table.table)
