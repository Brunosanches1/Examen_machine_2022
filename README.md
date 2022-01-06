# Examen_machine_2022
Examen machine du 6 Janvier 2022 : compression d'image parallèle par série de Fourier

**Dans le Makefile, modifier les premières lignes en commentant ou décommentant pour prendre le fichier adapter à votre environnement !**

__Pour ceux en distanciel__ : Pour toute question, envoyer un mail à xjuvigny@gmail.com (plus rapide que juvigny@onera.fr)

# Résultats
taux = 0.1

Séquentiel
| Type        | DFT                 | Compression | Reconstituition            |
| ----------- | ------------------- | ----------- | -------------------------- |
| Séquentiel  | 141775ms            | 8 ms        | 18419ms                    |    
| OpenMP      | 80259ms (acc=1.76)  | ----------- | 9534ms (acc=1.93)          |
| MPI - 1     | 79864 (acc=1.77)    | ----------- | 9923ms (acc=1.85)          |
| MPI - 2     |