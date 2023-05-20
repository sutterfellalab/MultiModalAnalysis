# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:03:33 2023

@author: Tim Kodalle
"""

import tkinter as tk


def inputGUI(inputDict, DictEntry, numberOfInputs, Title, Labels, TextPrompt):
            
    entries = []
    
    # Create a function to update the variables and close the GUI window
    def updateVariables():
        allEntries = []
        for entry in entries:   
            allEntries.append(entry.get())
        
        root.quit()
        root.destroy()  # close the GUI window
        
        # update class variable
        inputDict[DictEntry] = allEntries

    # Create the GUI window
    root = tk.Tk()
    root.title(Title)
    
    label = tk.Label(root, text=TextPrompt)
    label.grid(row=0, column=0, columnspan = 2)
    
    # Create the input fields and labels
    for i in range(numberOfInputs):
        label = tk.Label(root, text=Labels[i])
        label.grid(row=i+1, column=0, pady=10, padx=5)
        entry = tk.Entry(root)
        entry.grid(row=i+1, column=1, pady=10, padx=5)
        entries.append(entry)


    # Create a button to update the variables
    update_button = tk.Button(root, text="Submit", command=updateVariables)
    update_button.grid(row=numberOfInputs + 2, column=0, columnspan=2)
    
    # Start the GUI event loop
    root.mainloop()
    
    return


def selectionGUI(inputDict, DictEntry, title, options):
    
    root = tk.Tk()
    root.title(title)

    v = tk.IntVar(root)
    
    for i, option in enumerate(options):
        radioButton = tk.Radiobutton(root, text=option, variable=v, value=i)
        radioButton.grid(row=i, column=1)
        
    def submitButton():
        root.quit()
        root.destroy()
        
    submitButton = tk.Button(root, text="Submit", command=submitButton)
    submitButton.grid(row=len(options) + 2, column=1)
    
    root.mainloop()
    
    inputDict[DictEntry] = options[v.get()]
    
    return

def combinedGUI(inputDict, DictEntry, DictEntry2, DictEntry3, numberOfInputs, Title, Labels, TextPrompt, options, options2):
            
    entries = []
    boxes = []
    boxes2 = []
    
    # Create a function to update the variables and close the GUI window
    def updateVariables():
        allEntries = []
        allBoxes = []
        allBoxes2 = []
        for entry in entries:   
            allEntries.append(float(entry.get()))
            
        for box in boxes: 
            allBoxes.append(box.get())
        
        for box in boxes2: 
            allBoxes2.append(box.get())
            
        root.quit()
        root.destroy()  # close the GUI window
        
        # update class variable
        inputDict[DictEntry] = allEntries
        inputDict[DictEntry2] = allBoxes
        inputDict[DictEntry3] = allBoxes2

    # Create the GUI window
    root = tk.Tk()
    root.title(Title)
    
    label = tk.Label(root, text=TextPrompt)
    label.grid(row=0, column=0, columnspan = 2)
   
    # Create the input fields and labels
    for i in range(numberOfInputs):
        label = tk.Label(root, text=Labels[i])
        label.grid(row=i+1, column=0, pady=10, padx=5)
        entry = tk.Entry(root)
        entry.grid(row=i+1, column=1, pady=10, padx=5)
        entries.append(entry)
        
    for i, option in enumerate(options):
        v = tk.IntVar(root)
        checkBox = tk.Checkbutton(root, text=option, variable=v, onvalue=1, offvalue=0, command=None)
        checkBox.grid(row=i+1, column=3)
        boxes.append(v)
        
    for i, option in enumerate(options2):
        v = tk.IntVar(root)
        checkBox = tk.Checkbutton(root, text=option, variable=v, onvalue=1, offvalue=0, command=None)
        checkBox.grid(row=i+1, column=4)
        boxes2.append(v)

    # Create a button to update the variables
    update_button = tk.Button(root, text="Submit", command=updateVariables)
    update_button.grid(row=numberOfInputs + 2, column=0, columnspan=2)
    
    # Start the GUI event loop
    root.mainloop()
    
    return