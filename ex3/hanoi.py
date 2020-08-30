import sys
from itertools import combinations


def write_actions(domain_file, disks, pegs):
    curr_disks = disks.copy()
    for i in range(len(disks)):
        curr_disks.remove(disks[i])
        for comb in [list(x) for x in combinations(curr_disks + pegs, 2)]:
            domain_file.write("Name: " + disks[i] + "_from_" + comb[0] + "_to_" + comb[1] + "\n")
            domain_file.write("pre: " + disks[i] + " " + comb[1] + " " + disks[i] + comb[0] + "\n")
            domain_file.write("add: " + comb[0] + " " + disks[i] + comb[1] + "\n")
            domain_file.write("delete: " + comb[1] + " " + disks[i] + comb[0] + "\n")
            domain_file.write("Name: " + disks[i] + "_from_" + comb[1] + "_to_" + comb[0] + "\n")
            domain_file.write("pre: " + disks[i] + " " + comb[0] + " " + disks[i] + comb[1] + "\n")
            domain_file.write("add: " + comb[1] + " " + disks[i] + comb[0] + "\n")
            domain_file.write("delete: " + comb[0] + " " + disks[i] + comb[1] + "\n")


def create_domain_file(domain_file_name, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    domain_file = open(domain_file_name, 'w')  # use domain_file.write(str) to write to domain_file
    domain_file.write("Propositions:\n")
    for p in range(len(pegs)):
        domain_file.write(pegs[p] + ' ')
    for d in range(len(disks)):
        domain_file.write(disks[d] + ' ')
        for p in range(len(pegs)):
            domain_file.write(disks[d] + pegs[p] + ' ')
    for i in range(len(disks)):
        for j in range(i + 1, len(disks)):
            domain_file.write(disks[i] + disks[j] + ' ')
    domain_file.write("\nActions:\n")
    write_actions(domain_file, disks, pegs)

    domain_file.close()


def create_problem_file(problem_file_name_, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    problem_file = open(problem_file_name_, 'w')  # use problem_file.write(str) to write to problem_file
    problem_file.write("Initial state: ")
    for p in pegs:
        if p == 'p_0':
            continue
        problem_file.write(p + " ")
    problem_file.write(disks[0] + " ")
    for d in range(len(disks)):
        if d == len(disks) - 1:
            problem_file.write(disks[d] + "p_0\n")
            continue
        problem_file.write(disks[d] + disks[d + 1] + " ")
    problem_file.write("Goal state: ")
    for p in range(len(pegs) - 1):
        problem_file.write(pegs[p] + " ")
    problem_file.write(disks[0] + " ")
    for d in range(len(disks)):
        if d == len(disks) - 1:
            problem_file.write(disks[d] + pegs[-1] + "\n")
            continue
        problem_file.write(disks[d] + disks[d + 1] + " ")
    problem_file.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: hanoi.py n m')
        sys.exit(2)

    n = int(float(sys.argv[1]))  # number of disks
    m = int(float(sys.argv[2]))  # number of pegs

    domain_file_name = 'hanoi_%s_%s_domain.txt' % (n, m)
    problem_file_name = 'hanoi_%s_%s_problem.txt' % (n, m)

    create_domain_file(domain_file_name, n, m)
    create_problem_file(problem_file_name, n, m)
