from libc.stdint cimport int32_t, uint64_t
from mpi4py.libmpi cimport MPI_Comm, MPI_Datatype, MPI_Status, MPI_Op

cdef int abort_on_error(int ierr, MPI_Comm comm, unicode mpi_op) nogil

cdef void mpi_send(void* sendbuf, int32_t nitems, MPI_Datatype dtype,
                   int32_t destination, int32_t tag, MPI_Comm comm, void* token) nogil

cdef void mpi_recv(void* recvbuf, int32_t nitems, MPI_Datatype dtype, int32_t source,
                   int32_t tag, MPI_Comm comm, MPI_Status* status, void* token) nogil

cdef void mpi_sendrecv(void* sendbuf, int32_t sendcount, MPI_Datatype sendtype, int32_t dest, int32_t sendtag,
                       void* recvbuf, int32_t recvcount, MPI_Datatype recvtype, int32_t source, int32_t recvtag,
                       MPI_Comm comm, MPI_Status* status, void* token) nogil

cdef void mpi_allreduce(void* sendbuf, void* recvbuf, int32_t nitems,
                        MPI_Datatype dtype, MPI_Op op, MPI_Comm comm, void* token) nogil
