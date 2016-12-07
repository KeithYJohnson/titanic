def impute_data(data):
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna(
      data['Embarked'].value_counts().idxmax()
    )

    # Convert possible Embarked values to integers
    data["Embarked"][data["Embarked"] == "S"] = 0
    data["Embarked"][data["Embarked"] == "C"] = 1
    data["Embarked"][data["Embarked"] == "Q"] = 2

    # Convert genders to integers
    data["Sex"][data["Sex"] == "female"] = 1
    data["Sex"][data["Sex"] == "male"] = 0
    return data
