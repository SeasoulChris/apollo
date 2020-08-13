#!/usr/bin/env python3
"""
View functions for admin
"""

import flask

import application
from fueling.common import admin_utils
from utils import pwd_utils


blue_admin = flask.Blueprint("admin", __name__,
                             template_folder="templates",
                             static_url_path="static")
admin_db = admin_utils.AdminUtils()


@blue_admin.route("/admins", methods=["GET", "POST"])
def admins():
    """
    get admin list
    """
    admin_list = admin_db.get_admin_info({})
    return flask.render_template("admins.html", admin_list=admin_list,
                                 username=flask.session.get("user_info").get("username"))


@blue_admin.route("/create_admin", methods=["GET", "POST"])
def create_admin():
    """
    create admin by super user
    """
    username = flask.session.get("user_info").get("username")
    super_user = application.app.config.get("SUPER_USER_NAME")
    if username != super_user:
        return flask.render_template("error.html", error="不是超级用户没有权限",
                                     username=username)
    selected_role = flask.request.form.get("role_selected")
    role_type = list(application.app.config.get("ROLE_TYPE").keys())
    if flask.request.method == 'GET':
        return flask.render_template("admin.html", current_role=selected_role, role_type=role_type,
                                     username=username)
    admin_name = flask.request.form.get("admin_name")
    default_password = application.app.config.get("ADMIN_PASSWORD")
    email = flask.request.form.get("email")
    admin_db.save_admin_info(admin_name, pwd_utils.pwd_md5(default_password), selected_role, email)
    return flask.render_template("error.html", error="创建成功",
                                 username=username)


@blue_admin.route('/login', methods=["GET", "POST"])
def login():
    """
    login function
    """
    if flask.request.method == 'GET':
        return flask.render_template("login.html")
    role_dict = application.app.config.get("ROLE_TYPE")
    username = flask.request.form.get("user")
    hash_password = pwd_utils.pwd_md5(flask.request.form.get("pwd"))

    admin_obj = admin_db.db.find_one({"username": username, "password": hash_password})
    if not admin_obj:
        return flask.render_template("login.html", msg="用户名/密码错误")
    user_info = {
        "username": username,
        "email": admin_obj["email"],
        "role": admin_obj["role"],
        "permission": role_dict[admin_obj["role"]]
    }
    flask.session["user_info"] = user_info
    return flask.redirect('/jobs')


@blue_admin.route('/logout', methods=["GET", "POST"])
def logout():
    """
    logout function
    """
    flask.session.clear()
    return flask.redirect('/login')


@blue_admin.route("/reset_pwd", methods=["GET", "POST"])
def reset_pwd():
    """
    reset password function
    """
    username = flask.session.get("user_info").get("username")
    if flask.request.method == 'GET':
        return flask.render_template("reset.html", username=username)
    old_password = pwd_utils.pwd_md5(flask.request.form.get("old_pwd"))
    admin_obj = admin_db.get_admin_info({"username": username, "password": old_password})
    if not admin_obj:
        return flask.render_template("reset.html", msg="旧密码不正确",
                                     username=username)
    new_password = pwd_utils.pwd_md5(flask.request.form.get("new_pwd"))
    verify_password = pwd_utils.pwd_md5(flask.request.form.get("verify_pwd"))
    if new_password != verify_password:
        return flask.render_template("reset.html", msg="两次密码不一致",
                                     username=username)
    re = admin_db.db.update_one({"username": username}, {"$set": {"password": new_password}})
    if re.modified_count == 1:
        error = "密码修改成功"
    else:
        error = "密码修改失败"
    return flask.render_template("error.html", error=error,
                                 username=username)


@blue_admin.route("/allocate_permission", methods=["GET", "POST"])
def allocate_permission():
    """
    allocate permission for the role
    """
    urls = []
    for i in application.app.url_map.iter_rules():
        urls.append(i.rule)
    roles = application.app.config.get("ROLE_TYPE")
    if flask.request.method == 'POST':
        add_urls = flask.request.form.getlist("urls")
        role = flask.request.form.get("role")
        roles[role].extend(add_urls)
    return flask.render_template("permission.html", roles=roles, urls=urls,
                                 username=flask.session.get("user_info").get("username"))
