#from_month = "2014-05"
#to_month = "2017-07"

def interpolate_months(from_month,to_month):
    parts_from_month = from_month.split("-")
    interpolated_from_months = range(int(parts_from_month[1]),13)
    parts_to_month = to_month.split("-")
    interpolated_to_months = range(1,int(parts_to_month[1])+1)
    interpolated_years = range(int(parts_from_month[0])+1, int(parts_to_month[0])) #without to and for years

    list_of_interpolated_months = []

    for month in interpolated_from_months:
        list_of_interpolated_months.append(str(parts_from_month[0]) + "-" + str(month))

    for year in interpolated_years:
        for month in range(1,13):
            list_of_interpolated_months.append(str(year) + "-" + str(month))

    for month in interpolated_to_months:
        list_of_interpolated_months.append(str(parts_to_month[0]) + "-" + str(month))

    for month in list_of_interpolated_months:
        print(month)
    return list_of_interpolated_months


#interpolate_months(from_month,to_month)