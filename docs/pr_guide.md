# Code submission process
## Fork and clone code
1, Go to [PowerVR Paddle Model home page](https://github.com/imaginationtech/PaddlePaddle_Model_zoo), and click the fork button to generate a repository in your own directory, such as https://github.com/USERNAME/powervr_paddle_model.
2, Clone the remote repository to the local
```
# Pull the code of the develop branch
git clone https://github.com/USERNAME/PaddlePaddle_Model_zoo.git -b develop
cd powervr_paddle_model
```

## Establish a connection with upstream repository
we will create a remote connection to the original powervr_paddle_model repository and name it upstream.
```
git remote add upstream https://github.com/imaginationtech/PaddlePaddle_Model_zool
```

## Create a local branch
You can also create a new branch based on upstream branch, the command is as follows:
```
git checkout -b new_branch upstream/develop
```

## Modify and submit code
Assume you make change to README.md, and hope to submit it. 
```
git add README.md
git commit -s -m "you commit info"
```

## Keep the local repository up to date
Get the latest code from upstream and update the current branch.
```
git fetch upstream
git pull upsteram develop
```

## push to remote repository
```
git push origin new_branch
```

## Submit a Pull Request
Go to https://github.com/imaginationtech/PaddlePaddle_Model_zoo. Click new pull request, select the local branch and target branch, as shown in the figure below. In the description of the PR, fill in the functions completed by the PR. Next, wait for the review. If there is any need to modify, follow the steps above to update the corresponding branch in origin.