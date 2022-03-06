import df_generator
import time

def run_df_generator(train_path, test_path, n, save_path):
    start = time.time()
    df_generator.run(["--train-file", train_path, "--test-file", test_path, "--n", n, "--save-path", save_path], standalone_mode = False)
    print(f"Elapsed time: {round(time.time() - start, 2)}")


if __name__ == "__main__":
    run_df_generator("F:/work/kutatas/dwf/dwf_now/DeepWaterFramework/DeepBugHunter/tests/res/train.csv", "F:/work/kutatas/dwf/dwf_now/DeepWaterFramework/DeepBugHunter/tests/res/test.csv", 10, "testing")
    
