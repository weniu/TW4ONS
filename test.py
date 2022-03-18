class A():
    def __init__(self,b=1,c=0):
        self.a=10
        self.b=b
        self.c=c

    def fun1(self):
        self.a=20
        print(self.a)

    def fun2(self):
        self.fun1()

class B(A):
    def __init__(self,b=3,c=4):
        super().__init__(b,c)
        # self.b=b
        # self.c=c

    def fun1(self):
        self.a=-10
        print(self.a)

class C(A):
    def __init__(self):
        super().__init__()

B=B()
print(B.b,B.c)
C=C()
print(C.b,C.c)
