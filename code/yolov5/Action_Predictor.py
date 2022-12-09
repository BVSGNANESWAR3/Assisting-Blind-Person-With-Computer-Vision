#!/usr/bin/env python
# coding: utf-8

# In[ ]:
    # Audio specs
# import pyttsx3
# engine = pyttsx3.init() 
# engine.setProperty('rate', 100)  


# engine.setProperty('voice', 'english')
def flag_or(pflags,flags):
    for i in range(3):
        pflags[i]=flags[i]|pflags[i]
    return pflags
def flag_per_object(x,w,flags,pflags):
    f=[[0,0.40],[0.40,0.60],[0.60,1]]
    pl=x-w/2
    pr=x+w/2
    for i in range(3):
        t=0
        if(pflags[i]==1):
            t=0.01
        if(f[i][1]+t >=pl and f[i][0]-t <=pr):
            flags[i]=1
    
    return flags

def action(flags):
    if(flags[1]==0):
        return "straight"
    elif (flags[0]==0):
        return "left"
    elif(flags[2]==0):
        return "right"
    else:
        return "stop"

def text_output(PA,A,txt_path2):
    d={"straight":0,"left":1,"right":2,"stop":3}
    T=[["","straight","straight","straight"],["keep left","","keep left","keep left"],["keep right","keep right","","keep right"],["stop","stop","stop",""]]
#     audio_output(T[d[A]][d[PA]]);
    sen=T[d[A]][d[PA]]
    with open(f'{txt_path2}.txt', 'a') as f:
        f.write(T[d[A]][d[PA]]+"\n")
    PA=A
    return sen
    
# def audio_output(sen):
#     engine.say(sen)
#     engine.runAndWait()
       