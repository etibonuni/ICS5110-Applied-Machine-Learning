import pandas as pd

df = pd.read_csv("drug_consumption.csv")
#print df.iloc[0:10]

def summarizeClass(classLevel):
    switcher = {
        "CL0" : "CLEAN",
        "CL1" : "CLEAN",
        "CL2" : "CLEAN",
        "CL3" : "USER",
        "CL4" : "USER",
        "CL5" : "USER",
        "CL6" : "USER"
    }
    return switcher.get(classLevel, "CLEAN")

print(summarizeClass("CL0"))

for col in range(13, len(df.columns)):
    newColName = df.columns.values[col] + "_agg"
    print newColName
    df[newColName] = df.iloc[:,col].apply(summarizeClass)


print df.iloc[0:10]

df.to_csv("drug_consumption_binary.csv")