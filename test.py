def a(*args):
    print(type(args))
    for i in args:
        print(i)
a(1,2,3,4)