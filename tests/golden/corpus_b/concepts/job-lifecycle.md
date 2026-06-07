# Job lifecycle

A job moves through several states from creation to completion:
queued, running, succeeded, and failed. The scheduler promotes a
queued job to running when a worker picks it up, then records the
terminal state. Observers can subscribe to state transitions.
