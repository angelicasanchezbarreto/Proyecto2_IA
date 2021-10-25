import matplotlib.pyplot as plt

font1 = {'family':'serif','size':20}
font2 = {'family':'serif','size':15}

def plot_error(epochs,error_train_list,error_valid_list,name):
    plt.figure()
    p1 = plt.scatter(x=list(range(epochs)), y=error_train_list,color="lightskyblue")
    p2 = plt.scatter(x=list(range(epochs)), y=error_valid_list,color="lightgreen")
    
    plt.legend((p1,p2),["Train error","Validation error"])
    
    plt.title("Train and Validation Error vs. Epoch",fontdict=font1)
    plt.xlabel("Epoch",fontdict=font2)
    plt.ylabel("Error",fontdict=font2)
    #plt.show()
    plt.savefig("plots/plot_"+name+".png")

def plot_error2(epochs,error_train_list,error_valid_list,error_test_list,name):
    plt.figure()
    p1 = plt.scatter(x=list(range(epochs)), y=error_train_list,color="lightskyblue",alpha=0.75)
    p2 = plt.scatter(x=list(range(epochs)), y=error_valid_list,color="lightgreen",alpha=0.75)
    p3 = plt.scatter(x=list(range(epochs)), y=error_test_list,color="mediumpurple",alpha=0.75)
    
    plt.legend((p1,p2,p3),["Train error","Validation error","Test error"])

    plt.title("Train, Validation and Test Error vs. Epoch",fontdict=font1)
    plt.xlabel("Epoch",fontdict=font2)
    plt.ylabel("Error",fontdict=font2)
    #plt.show()
    plt.savefig("plots/plot_"+name+".png")
    
def save_files(alpha_nums,const_nums):
    alpha_file = open("files/alpha_nums.txt", "w")
    const_file = open("files/const_nums.txt", "w")
    for i in range(len(alpha_nums)):
        alpha_file.write(str(alpha_nums[i]) + "\n")
        const_file.write(str(const_nums[i]) + "\n")

    alpha_file.close()
    const_file.close()