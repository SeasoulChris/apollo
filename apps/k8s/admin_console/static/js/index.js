$(document).ready(function () {

    // show the message
    if (widget.messageBar.init()){
         $(document).on("widget.mb.show",function(e, options){
             widget.messageBar.show(options);
         });
    }

    // tooltip
    $('.failure-code').tooltip({container: 'body'});
    $('.td-job-id').tooltip({container: 'body'});

    // change the edit checkbox
    $("#package-div input").change(function(){
        switchSelect(this);
    });

    // hide the edit modal
    $('#editModal').on('hide.bs.modal', function (e) {
        var job_check = $("#editModal").children().children().children(".modal-body").children().first("div");
        job_check.empty();
        var inputDom = $("#package-div input");
        inputDom.removeAttr("checked");
        inputDom.val("Disabled");
    })

   // show the edit modal
    $('#editModal').on('show.bs.modal', function (e) {
        var button = $(e.relatedTarget);
        var account_services = eval(button.data("account-services"));
        var account_data = {};
        var account_body = $("#editModal").children().children().children(".modal-body")
        var account_body_job = account_body.children().first("div");
        var account_body_check = account_body.children().last("div");
        account_data["account_id"] = button.data("account-id");

        var old_status_dict = {};
        var chose_input_dom = $("#package-div").children("input");
        old_status_dict[chose_input_dom.attr("name")] = chose_input_dom.val();
        for (j = 0; j<account_services.length; j++){
            var show_job_type = account_services[j]["job_type"];
            var show_used = 0;
            if (account_services[j].used){
                show_used = account_services[j]["used"]
            }
            var show_dom;
            var account_service_status = account_services[j]["status"]
            old_status_dict[show_job_type] = account_service_status;
            if (account_service_status == "Enabled"){
                show_dom = "<div style={'display': 'block'}><input name='"+show_job_type+"' type='checkbox' checked value="
                 + account_service_status  + "><span>" + showJobType(show_job_type) + "    "+ "使用:" + show_used+ "</span></div>"
            }
            else{
                show_dom = "<div style={'display': 'block'}><input name='"+show_job_type+"' type='checkbox' value="
                 + account_service_status  + "><span>" + showJobType(show_job_type) + "    "+ "使用:" + show_used+ "</span></div>"
            }
            account_body_job.append(show_dom);
        };

        account_body_check.css("display", "block");
        $(".edit_quota").one("click", function(e){
            setCheckboxValue(account_body_job.children());
            checkBoxIsChecked(chose_input_dom);
            var new_status_dict = getObjDict(account_body_job.children());
            new_status_dict[chose_input_dom.attr("name")] = chose_input_dom.val();

            if(!cmp(old_status_dict, new_status_dict)){

                // when the checkbox is changed
                if (new_status_dict["chose_package"] == "Enabled"){
                    var package_selected = $("#package-select option:selected").val();
                }
                account_data["service_package"] = package_selected;
                for(var key in new_status_dict){
                    if(key != "chose_package"){
                        account_data[key] =  new_status_dict[key];
                    }
                }
                $.ajax({
                    url: "/api/v1/namespaces/default/services/http:admin-console-service:8000/proxy/edit_quota",
                    dataType: "json",
                    type: "POST",
                    data: account_data,
                    success: function (result) {

                        // hide the editModal
                        $("#editModal").modal("hide");

                        // get the data from flask
                        var account_data = result["data"];
                        var service_dom = $("#service-tr-"+account_data["_id"]).nextUntil("#operation-tr-"+account_data["_id"])
                        var services = account_data["services"]

                        // update the services list
                        for (i=0; i<service_dom.length; i++){
                            $(service_dom[i]).children().text(showJobType(services[i]["job_type"])+":"+showServiceStatus(services[i]["status"])+"  使用:"+services[i]["used"]);
                        };

                        // update the used and remaining quota
                        $("#used-"+account_data["_id"]).text(account_data["used"]);
                        $("#remaining-"+account_data["_id"]).text(account_data["remaining_quota"]);

                        // update the due_date
                        $("#due-date-"+account_data["_id"]).text(account_data["due_date"]);

                        // update the dialog data-account-services attr
                        $("#edit-action-"+account_data["_id"]).attr("data-account-services", services);

                        // update the button data
                        button.data("account-services", services);

                        // update the message
                        if (new_status_dict["chose_package"] == "Enabled"){
                            $(document).trigger("widget.mb.show", {type:"ok",message:"用户（邮箱："+account_data["com_email"]+"）剩余配额"+account_data["remaining_quota"]});
                        };
                    }
                })
            } else {
                $("#editModal").modal("hide");
            }
        })
    });

    // close update message
    $("#update-message-close").click(function () {
        $("#update-message").text("");
        $(this).css("display", "none");
    });

    // close search message
    $("#filter-message-close").click(function () {
        $("#sn-search").attr("value","");
        $("#filter-message").text("");
        $("#job-form").submit();
    });

    // submit the job-type select
    $('#type-select').change(function () {
        $("#job-form").submit();
    });

    // submit the time-field select
    $('#time-select').change(function () {
        $("#job-form").submit();
    });

    // submit the search by vehicle SN in job
    $("#search-btn").click(function () {
        $("#job-form").submit();
    });

    // submit the search by vehicle SN in account
    $("#account-search-btn").click(function () {
        $("#account-form").submit();
    });

    // clear the vehicle in account
    $("#account-message-close").click(function () {
        $("#account-sn-search").attr("value","");
        $("#account-filter-message").text("");
        $("#account-comment-text").text();
        $("#account-form").submit();
    });

    // close search message
    $("#statistics-filter-message-close").click(function () {
        $("#statistics-filter-message").text("");
        $("#statictics-form").submit();
    });

    // submit the job-type select
    $('#statistics-type-select').change(function () {
        $("#statictics-form").submit();
    });

    // submit the time-field select
    $('#statistics-time-select').change(function () {
        $("#statictics-form").submit();
    });

    // submit the search by vehicle SN
    $("#statistics-search-btn").click(function () {
        $("#statictics-form").submit();
    });

    // submit the aggregate-field select
    $('#aggregated-select').change(function () {
        $("#statictics-form").submit();
    });

    // show the modal
    $('#accountModal').on('show.bs.modal', function (e) {
        var button = $(e.relatedTarget);
        var account_data = {};
        var label_action = {
            "Enable": "开通",
            "Reject": "打回",
            "Disable": "停用"
        };

        var status_action = {
            "Enable": "Enabled",
            "Reject": "Rejected",
            "Disable": "Disabled"
        };

        var status_btn_action = {
            "Enabled": "Disable",
            "Rejected": "",
            "Disabled": "Enable"
        };

        // Get the user basic infomation from dialog
        account_data["label"] = button.data("label");
        account_data["verhicle_sn"] = button.data("verhicle_sn");
        account_data["com_name"] = button.data("com_name");
        account_data["com_email"] = button.data("com_email");
        account_data["bos_bucker_name"] = button.data("bos_bucker_name");
        account_data["bos_region"] = button.data("bos_region");
        account_data["account_status"] = button.data("account_status");
        account_data["account_id"] = button.data("account_id");
        account_data["action"] = status_action[account_data["label"]];
        first_status = button.data("first_status");
        var account_services = button.data("account-services");

        var modal = $(this);
        
        var str_label = label_action[account_data["label"]];
        var str_label_type = account_data["label"] == "Reject" ? "申请" : "服务";

        // Apply the chinese action
        modal.find('.modal-title').text(str_label + "用户" + str_label_type);
        modal.find('.action-label').text("您确定要" + str_label + "这个用户" + str_label_type + "吗？");
        modal.find('.verhicle-sn').text("车辆信息:" + account_data["verhicle_sn"]);
        modal.find('.com-name').text("公司名字:" + account_data["com_name"]);
        modal.find('.com-email').text("公司邮箱:" + account_data["com_email"]);
        modal.find('.bos-name').text("BOS名字:" + account_data["bos_bucker_name"]);
        modal.find('.bos-region').text("BOS区域:" + account_data["bos_region"]);
        modal.find('.account-status').text("账号状态:" + account_data["account_status"]);
        modal.find('.btn-account-action').text(str_label);

        // Submit the form for verification field and post ajax request
        $("form", this).first().one("submit", function(e){
            account_data["comment"] = $("#account-comment-text").val();
            $.ajax({
                url: "/api/v1/namespaces/default/services/http:admin-console-service:8000/proxy/update_status",
                dataType: "json",
                type: "POST",
                data: account_data,
                beforeSend: function () {
                    $("#account-comment-text").attr("disabled", "disabled");
                },
                success: function (result) {
                     // The action to be executed
                     var not_action = status_btn_action[account_data["action"]];
                     // The status to be set
                     var status = result["operation"]["status"].toString();

                     // Blank form
                     $("#account-comment-text").val("");
                     // Cancel the disabled
                     $("#account-comment-text").removeAttr("disabled");
                     // Hide modal box
                     $('#accountModal').modal("hide");

                     // Set the data-label attr
                     if (not_action.length !== 0)
                     {
                        $("#button-action-" + account_data["account_id"]).attr("data-label", not_action);
                        $("#button-action-" + account_data["account_id"]).attr("data-account_status", status);
                        // Set the lable of button data
                        button.data("label", not_action);
                        button.data("account_status", status);
                        if (status == "Enabled")
                        {
                            if(first_status == "Pending")
                            {
                                $("#button-reject-" + account_data["account_id"]).remove();
                                var dom = '<a style="width: 100%" id="edit-action-' + account_data["account_id"] +
                                        '" class="account-edit-modal" data-toggle="modal" data-target="#editModal" data-account-services="' +
                                        account_services + '" data-account-id="' + account_data["account_id"] + '"> Edit </a>' 
                                $("#button-action-" + account_data["account_id"]).after(dom);
                            }
                            else
                            {
                                $("#edit-action-" + account_data["account_id"]).css('display', 'inline');
                            }
                        }
                        else if (status == "Disabled")
                        {
                            $("#edit-action-" + account_data["account_id"]).css('display', 'none');
                        }
                     }
                     else
                     {
                        $("#button-action-" + account_data["account_id"]).css('display', 'none');
                        $("#button-reject-" + account_data["account_id"]).css('display', 'none');
                     }

                     // update the table
                     // Update the status
                     $("#status-" + account_data["account_id"]).text(status);
                     // Update the action
                     if (not_action.length !== 0)
                     {
                        $("#button-action-" + account_data["account_id"]).text(not_action);
                     }
                     // Update the action history
                     $("#flag-" + account_data["account_id"]).before('<tr style="display: none" class="operation_span">' +
                        '<td colspan="11" style="border-top: none">' +
                        "用户账号被" + result["operation"]["email"] + "在" + result["operation"]["time"] +
                        str_label + "。备注是：" + result["operation"]["comments"] + "。</td></tr>");

                     // Update the message
                     $(document).trigger("widget.mb.show", {type:"ok",message:"用户账号（车辆编号："+account_data["verhicle_sn"]+"）被"+str_label});

                     // Expand the history action
                     $("#expand-" + account_data["account_id"]).css("display", "none");
                     $("#collapse-" + account_data["account_id"]).css("display", "inline");
                     if (not_action.length !== 0)
                     {
                        // $("#button-" + account_data["label"].toLowerCase() + "-" + account_data["account_id"]).parent().parent().nextUntil(".account_flag").css("display", "table-row");
                        $("#button-action-" + account_data["account_id"]).parent().parent().nextUntil(".account_flag").css("display", "table-row");
                    }                    
                }
            });
            return false;
         })
        });

    // hide the modal
    $('#accountModal').on('hide.bs.modal', function (e) {
         $("#account-comment-text").val("");
    })

    // show the modal
    $('#myModal').on('show.bs.modal', function (e) {
        var button = $(e.relatedTarget);
        var job_data = {};
        var action_label = {
            "Invalid": "无效",
            "Valid": "有效"
        };
        var action_not_label = {
            "Invalid": "Valid",
            "Valid": "Invalid"
        };
        // Get the job-id and action from dialog
        job_data["job_id"] = button.data("job-id");
        job_data["action"] = button.data("job-action");

        // Apply the chinese action
        $("#myModalLabel").text("设置" + action_label[job_data["action"]]);
        $(".comment-title").text("你确认把当前任务设置为" + action_label[job_data["action"]] + "吗?");

        // Submit the form for verification field and post ajax request
        $("form", this).first().one("submit", function(e){
            job_data["comment"] = $("#comment-text").val();
            $.ajax({
                url: "/api/v1/namespaces/default/services/http:admin-console-service:8000/proxy/submit_job",
                dataType: "json",
                type: "POST",
                data: job_data,
                beforeSend: function () {
                    $("#comment-text").attr("disabled", "disabled");
                },
                success: function (result) {
                     // The action to be executed
                     var not_action = action_not_label[job_data["action"]];
                     // The is_valid to be set
                     var is_valid = result["operation"]["is_valid"].toString();
                     is_valid = is_valid.substring(0, 1).toUpperCase() + is_valid.substring(1); // 返回的is_valid

                     // Blank form
                     $("#comment-text").val("");
                     // Cancel the disabled
                     $("#comment-text").removeAttr("disabled");
                     // Hide modal box
                     $('#myModal').modal("hide");

                     // Set the data-job-action attr
                     $("#button-"+job_data["job_id"]).attr("data-job-action", not_action);
                     // Set the job-action of button data
                     button.data("job-action", not_action);

                     // update the table
                     // Update the is_valid
                     $("#is-valid-" + job_data["job_id"]).text(is_valid);
                     // Update the action
                     $("#button-" + job_data["job_id"]).text("设置"+action_label[not_action]);
                     // Update the action history
                     $("#flag-" + job_data["job_id"]).before('<tr style="display: none" class="operation_span">' +
                        '<td colspan="11" style="border-top: none">' +
                        "任务被" + result["operation"]["email"] + "在" + result["operation"]["time"] +
                        "设置为" + action_label[job_data["action"]] + "。备注是：" + result["operation"]["comments"] + "。</td></tr>");

                     // Update the message
                     $(document).trigger("widget.mb.show", {type:"ok",message:"任务（序号："+job_data["job_id"]+"）被设置为"+action_label[job_data["action"]]});

                     // Expand the history action
                     $("#expand-" + job_data["job_id"]).css("display", "none");
                     $("#collapse-" + job_data["job_id"]).css("display", "inline");
                     $("#button-" + job_data["job_id"]).parent().parent().nextUntil(".flag").css("display", "table-row");
                }
            });
            return false;
         })
        });

    // hide the modal
    $('#myModal').on('hide.bs.modal', function (e) {
         $("#comment-text").val("");
    })

    // Click the expand
    $(".job_expand").click(function () {
        $(this).css("display", "none");
        $(this).next().css("display", "inline");
        $(this).parent().parent().nextUntil(".flag").css("display", "table-row")
    })

    // Click the collapse
    $(".job_collapse").click(function () {
        $(this).css("display", "none");
        $(this).prev().css("display", "inline");
        $(this).parent().parent().nextUntil(".flag").css("display", "none")
    })

    // account click the expand
    $(".account_expand").click(function () {
        $(this).css("display", "none");
        $(this).next().css("display", "inline");
        $(this).parent().parent().nextUntil(".account_flag").css("display", "table-row")
    })

    // account click the collapse
    $(".account_collapse").click(function () {
        $(this).css("display", "none");
        $(this).prev().css("display", "inline");
        $(this).parent().parent().nextUntil(".account_flag").css("display", "none")
    })
});

var widget = {};
widget["messageBar"] = {};
widget.messageBar["init"] = function(){
    var domMB = $("#messageBar");
    if (!domMB ) return false;
    domMB.children().last().on("click",function(e){
        widget.messageBar.hide();
    })
    return true;
};

// hide event
widget.messageBar["hide"] = function(){
    var domMB = $("#messageBar");
    if (!domMB ) return false;
    domMB.css("display","none");
    return true;
};

// show event
widget.messageBar["show"] = function(options){
    var domMB = $("#messageBar");
    if (!domMB || !options) return false;
    domMB.children().first().text(options.message);
    domMB.css("display", "block");
    switch (options.type) {
      case "ok":
        // set ok icon
        break;
      case "error":
        // set error icon
        break;
      case "warning":
        // set error icon
        break;
    }
    return true;
};

// switch the select disabled attr
function switchSelect(input) {
    if($(input).is(":checked")){
        $("#package-select").removeAttr("disabled");
    }
    else{
        $("#package-select").attr("disabled",true);
    }
};

// show the job type in the page
function showJobType(job_type) {
    return job_type.split("_").map(function(item, index) {
        return item.slice(0, 1).toUpperCase() + item.slice(1);
    }).join(' ');
}

// show the status in the page
function showServiceStatus(status){
    var status_dict = {
        "Enabled": "启用",
        "Pending": "待审批",
        "Rejected": "驳回",
        "Disabled": "停用",
        "Expired": "过期",
        "Over-quota": "超额"
    }
    return status_dict[status]
}

// Get the formObj from form
function getObjDict(objs) {
    var formObj = {};
    for(i=0; i<objs.length; i++){
        var input_dom = $(objs[i]).children("input");
        formObj[input_dom.attr("name")] = input_dom.val()
    }
    return formObj;
}

// Circulation set the value in checkbox
function setCheckboxValue(objs){
    for(i=0; i<objs.length; i++){
        var input_dom = $(objs[i]).children("input");
        checkBoxIsChecked(input_dom);
    };
}

// Set the value in checkbox
function checkBoxIsChecked(dom){
    if(dom.is(":checked")){
        dom.val("Enabled");
    }else{
        dom.val("Disabled");
    }
}

// Compare two objects for equality
function cmp( x, y ) {
    if ( x === y ) {
        return true;
    }
    if ( ! ( x instanceof Object ) || ! ( y instanceof Object ) ) {
        return false;
    }
    if ( x.constructor !== y.constructor ) {
        return false;
    }
    for ( var p in x ) {
        if ( x.hasOwnProperty( p ) ) {
            if ( ! y.hasOwnProperty( p ) ) {
                return false;
            }
            if ( x[ p ] === y[ p ] ) {
                continue;
            }
            if ( typeof( x[ p ] ) !== "object" ) {
                return false;
            }
            if ( ! Object.equals( x[ p ],  y[ p ] ) ) {
                return false;
            }
        }
    }
    for ( p in y ) {
        if ( y.hasOwnProperty( p ) && ! x.hasOwnProperty( p ) ) {
            return false;
        }
    }
    return true;
}