class SEIRSimulator:

    def __init__(
        self,
        population,
        initial_infected,
        beta=0.30,
        sigma=0.20,
        gamma=0.10
    ):
        self.population = population

        self.S = population - initial_infected
        self.E = 0
        self.I = initial_infected
        self.R = 0

        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma

    def simulate_day(
        self,
        vaccine_doses=0,
        testing_kits=0,
        staff=0
    ):

        # Vaccination
        vaccinated = min(vaccine_doses, self.S)

        self.S -= vaccinated
        self.R += vaccinated

        # Testing reduces transmission
        testing_effect = min(testing_kits / self.population, 0.30)

        effective_beta = self.beta * (1 - testing_effect)

        # Medical staff improves recovery
        staff_effect = min(staff * 0.002, 0.20)

        effective_gamma = self.gamma + staff_effect

        # SEIR equations
        new_exposed = effective_beta * self.S * self.I / self.population

        new_infected = self.sigma * self.E

        new_recovered = effective_gamma * self.I

        # Safety limits
        new_exposed = min(new_exposed, self.S)
        new_infected = min(new_infected, self.E)
        new_recovered = min(new_recovered, self.I)

        # Update compartments
        self.S -= new_exposed
        self.E += new_exposed - new_infected
        self.I += new_infected - new_recovered
        self.R += new_recovered

        return {
            "susceptible": self.S,
            "exposed": self.E,
            "infected": self.I,
            "recovered": self.R
        }


def run_simulation(
    population,
    initial_infected,
    days=60,
    vaccine_doses=0,
    testing_kits=0,
    staff=0
):

    simulator = SEIRSimulator(
        population=population,
        initial_infected=initial_infected
    )

    history = []

    for day in range(days):

        result = simulator.simulate_day(
            vaccine_doses=vaccine_doses,
            testing_kits=testing_kits,
            staff=staff
        )

        result["day"] = day + 1

        history.append(result)

    return history


def compare_interventions():

    population = 1_000_000
    initial_infected = 500
    days = 60

    # Without intervention
    no_intervention = run_simulation(
        population=population,
        initial_infected=initial_infected,
        days=days
    )

    # With intervention
    intervention = run_simulation(
        population=population,
        initial_infected=initial_infected,
        days=days,
        vaccine_doses=200,
        testing_kits=1000,
        staff=20
    )

    # Total infected over all days
    total_no_intervention = sum(
        day["infected"]
        for day in no_intervention
    )

    total_intervention = sum(
        day["infected"]
        for day in intervention
    )

    cases_prevented = (
        total_no_intervention
        - total_intervention
    )

    reward = cases_prevented / total_no_intervention

    print("\nSEIR SIMULATION RESULTS")
    print("------------------------")

    print(
        f"Total infected without intervention: "
        f"{total_no_intervention:.0f}"
    )

    print(
        f"Total infected with intervention: "
        f"{total_intervention:.0f}"
    )

    print(
        f"Cases prevented: "
        f"{cases_prevented:.0f}"
    )

    print(
        f"Reward: "
        f"{reward:.4f}"
    )


if __name__ == "__main__":
    compare_interventions()