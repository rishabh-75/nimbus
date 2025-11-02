class BankAccount:
    def __init__(self, account_number, balance):
        self._account_number = account_number
        self.__balance = balance

    # getter method for the private attributes
    def get_balance(self):
        return self.__balance

    # setter methor for the private attribute
    def set_balance(self, balance):
        if balance >= 0:
            self.__balance = balance
        else:
            print("!! Invalid balance")

    # public method that uses the private attributes
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
        else:
            print("!! Invalid deposit amount")

    # public method that uses the private attributes
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
        else:
            print("!! Invalid withdrawal amount")


account = BankAccount("3249", 10000)
print("Account Number: ", account._account_number)
account.set_balance(500)
print("Updated Balance: ", account.get_balance())
