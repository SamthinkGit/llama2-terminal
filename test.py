string = """"asdfasdfasdf
asdfasdf
asdfa
sdfasdf
AI:asdfasdf
asdf
AI: Hola mi nombre
[asdfasdfa]as"""

lineas = string.split('\n')

for linea in lineas:
    if linea.startswith('AI:'):
        last = linea

print(last)
