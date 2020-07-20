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

    // select drop-down box
    $('select').selectpicker();

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
