import config

def estimatePrice(km, theta0=config.theta0, theta1=config.theta1):
    return theta0 + theta1 * km

def main():
    while True:
        try:
            print(estimatePrice(int(input("Car km ? "))))
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()
