# ps -ef|grep python | tr -s ' '|cut -d' ' -f2 | xargs kill -9

# tensorboard --logdir tensorboard


# for i, A_x in enumerate(model.sp_A):
#     with open('./tu_{}.txt'.format(i),'w') as f:
#         for i in A_x:
#             for j in i:
#                 f.write(str(j))
#                 f.write(' ')
#             f.write('\n')