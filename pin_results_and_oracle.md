# Oracle

## A-D-64QFP-14B-SM.png

Bad: L[0], R[15], B[0]

Ambigious: T[0], T[2-3], L[15]

## A-J-28SOP-01B-SM.png

Bad: T[15], L[6], B[0]

Ambigious: R[15]

## A-D-64QFP-15B-SM.png

Bad: B[0]

Ambigious: B[13]

## C-T-28SOP-04F-SM.png

Bad: B[13]

Ambigious: B[2-3]

# Results

## Results table where Ambigious counted as Correct

|                | A-D-64QFP-14B-SM.png | A-J-28SOP-01B-SM.png | A-D-64QFP-15B-SM.png | C-T-28SOP-04F-SM.png |
| -------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| True Positive  | 6                    | 4                    | 2                    | 3                    |
| False Positive | 3                    | 0                    | 1                    | 0                    |
| True Negative  | 55                   | 60                   | 25                   | 25                   |
| False Negative | 0                    | 0                    | 0                    | 0                    |
| Accuracy       | 61/64 = 95.3%        | 64/64 = 100%         | 27/28 = 96.4%        | 28/28 = 100%         |

Overall Accuracy: 180/184 = 97.8%

## Results table where Ambigious counted as Incorrect

|                | A-D-64QFP-14B-SM.png | A-J-28SOP-01B-SM.png | A-D-64QFP-15B-SM.png | C-T-28SOP-04F-SM.png |
| -------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| True Positive  | 3                    | 3                    | 1                    | 1                    |
| False Positive | 6                    | 1                    | 2                    | 2                    |
| True Negative  | 55                   | 60                   | 25                   | 25                   |
| False Negative | 0                    | 0                    | 0                    | 0                    |
| Accuracy       | 58/64 = 90.6%        | 63/64 = 98.4%        | 26/28 = 92.9%        | 26/28 = 92.9%        |

Overall Accuracy: 173/184 = 94.0%
