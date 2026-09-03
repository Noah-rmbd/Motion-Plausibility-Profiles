try:
    from .synthetic_anomalies import main
except ImportError:
    from synthetic_anomalies import main


if __name__ == "__main__":
    main()
