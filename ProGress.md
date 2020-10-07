1. preprocess OK
2. trainning set FEAT extract OK
3. test_audio OK
4. features:     SN = np.concatenate((SN1, SN2, SN3), axis=0) #(522, frames)
                SIN = np.concatenate((SIN1, SIN2, SIN3), axis=0) #(522, frames)
                ZN = np.concatenate((ZN1, ZN2, ZN3), axis=0) #(522, frames)
                SN_SIN_ZN = np.concatenate((SN, SIN, ZN), axis=0) #(1566, frames)
5. unfinished dataset(length issue) to check , some of label files are still dismatch its end time. OK
6. Model establish >loss >train -> label transform onoff adjust


LOSS should contain label transform info

7. put testdata to tensorboard OK
8. logger coding save model and code to runs
9. doing on utils.testset_evaluation
10. wholesongtest length unmatch issue