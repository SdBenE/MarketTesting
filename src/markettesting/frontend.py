import formatting
import modelCreation

def listHelp():
    print("format : Allows you to format your tickers to prevent empty data for model entry")
    print("model : allows you to create and import a stored model")

def checkInput(entry):
    if entry == 'help':
        listHelp()
    elif entry == 'format':
        entry2 = input('How many years would you like to format from? : ')

        if entry2 == int(entry2):
            formatting.fileFormation(entry2)
        else:
            print(f'[ERROR!] : Format {entry2} was not expected. Program expected entry of type <int>.')
    elif entry == 'model':
        entry2 = input('<MODEL> What woudld you like to do with said model? : ')

        if entry2 == 'create':
            entry3 = input('<MODEL CREATE> Enter the number of epochs and units for the model, separated by spaces: ')
            epochCount, unitCount = entry3.split()
            currentModel = modelCreation.LTSMModel(epochs=int(epochCount), units=int(unitCount))
    else:
        print(f'[ERROR!] : Entry {entry} was invalid!')


def intro():
    print(r"::::    ::::      :::     :::::::::  :::    ::: :::::::::: :::::::::::         ")
    print(r"+:+:+: :+:+:+   :+: :+:   :+:    :+: :+:   :+:  :+:            :+:             ")
    print(r"+:+ +:+:+ +:+  +:+   +:+  +:+    +:+ +:+  +:+   +:+            +:+             ")
    print(r"+#+  +:+  +#+ +#++:++#++: +#++:++#:  +#++:++    +#++:++#       +#+             ")
    print(r"+#+       +#+ +#+     +#+ +#+    +#+ +#+  +#+   +#+            +#+             ")
    print(r"#+#       #+# #+#     #+# #+#    #+# #+#   #+#  #+#            #+#             ")
    print(r"###       ### ###     ### ###    ### ###    ### ##########     ###             ")
    print(r"::::::::::: :::::::::: :::::::: ::::::::::: ::::::::::: ::::    :::  ::::::::  ")
    print(r"    :+:     :+:       :+:    :+:    :+:         :+:     :+:+:   :+: :+:    :+: ")
    print(r"    +:+     +:+       +:+           +:+         +:+     :+:+:+  +:+ +:+        ")
    print(r"    +#+     +#++:++#  +#++:++#++    +#+         +#+     +#+ +:+ +#+ :#:        ")
    print(r"    +#+     +#+              +#+    +#+         +#+     +#+ +#+# +#+   +#+# ")
    print(r"    #+#     #+#       #+#    #+#    #+#         #+#     #+#   #+#+# #+#    #+# ")
    print(r"    ###     ########## ########     ###     ########### ###    ####  ########  ")

    for x in range(0,5):
        print('\n')