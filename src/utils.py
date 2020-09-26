# util Functions

# Arguments tratment function
def getArgs(args, param):
    try:
        argValue = False
        for a in args:
            if(a == '-h'):
                print('============= Help Menu =============')
                print("Exec format: python3 main.py [args] | python main.py -h for help")
                print("\n\nCommands:")
                print('-i "datasetName" on data folder')
                return False
        for i in range(1, len(args)):
            if(args[i] == param):
                argValue = args[i+1]
        if(argValue == False): raise Exception("Param " + str(param) + " couldn't be found")
        return argValue
    except Exception as e:
        print("Exec format: python main.py [args] | python main.py -h for help.\nError: " + str(e))
        exit() 