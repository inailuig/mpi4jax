from mpi4py import MPI
import mpi4jax
import jax.numpy as jnp
import jax

MPI_comm = MPI.COMM_WORLD
n_nodes = MPI_comm.Get_size()
node_number = MPI_comm.Get_rank()

from jax.lax import create_token


m, n = 5,12
k1, k2, k3 = jax.random.split(jax.random.PRNGKey(123),3)

A = jax.random.uniform(k1,(m,n))
x = jax.random.uniform(k2,(n,))
y = jax.random.uniform(k2,(m,))

assert(n%n_nodes==0)
n_local = n//n_nodes
start_local = n_local*node_number
end_local = start_local+n_local

A_local = A[:, start_local:end_local]
x_local = x[start_local:end_local]


test_token = create_token(123)

def allreduce_sum(x):
    res, token = mpi4jax.Allreduce(x, op=MPI.SUM, comm=MPI_comm, token=test_token)
    return res

# essentially just an idenitity
def allreduce_sumT(y):
    res, token = mpi4jax.AllreduceT(y, op=MPI.SUM, comm=MPI_comm, token=test_token)
    return res


def matvec_mpi(A_local, x_local):
    return allreduce_sum(A_local@x_local)

def matvec_transpose_mpi(A_local, y_global):
    return A_local.T @ allreduce_sumT(y_global)

mv = lambda x_local: matvec_mpi(A_local, x_local)
mvT = lambda y: matvec_transpose_mpi(A_local, y)




print('Ax on rank', node_number, ': ', mv(x_local))

MPI_comm.Barrier()
print('ATy on rank', node_number, ': ', mvT(y))
MPI_comm.Barrier()


def transpose(f, x):
    return lambda y: jax.linear_transpose(f, x)(y)[0]


lt = transpose(mv, x_local)
print('lt Ax on rank', node_number, ': ', lt(y))
MPI_comm.Barrier()

ltT = transpose(mvT, y)
print('lt ATy on rank', node_number, ': ', ltT(x_local))
MPI_comm.Barrier()


ltlt = transpose(lt, y)
print('ltlt Ax on rank', node_number, ': ', ltlt(x_local))
MPI_comm.Barrier()


ltltT = transpose(ltT, x_local)
print('ltlt ATy on rank', node_number, ': ', ltltT(y))
MPI_comm.Barrier()


ltltlt = transpose(ltlt, x_local)
print('ltltl Ax on rank', node_number, ': ', ltltlt(y))
MPI_comm.Barrier()

ltltltT = transpose(ltltT, y)
print('ltltlt ATy on rank', node_number, ': ', ltltltT(x_local))
MPI_comm.Barrier()
