print("1 - add")
print("2 - subtract")
print("3 - multiply")
print("4 - divide")
option = int(input("choose an operation: "))

if(option in [1,2,3,4]):

    num1 = int(input("Enter a number: "))
    num2 = int(input("Enter a number: "))

    if (option == 1):
        result = num1 + num2

    elif (option == 2):
         result = num1 - num2

    elif (option == 3):
        result = num1 * num2

    if (option == 4):  
        result = num1 // num2

print("The result of the operation is {}".format(result))
    


     